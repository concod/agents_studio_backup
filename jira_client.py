"""
Jira API Client - Refactored for FastAPI integration
"""
import os
import re
from typing import Optional, Dict, Any, List
from urllib.parse import unquote
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()


class JiraClient:
    def __init__(self):
        self.site = os.getenv("JIRA_SITE", "").rstrip("/")
        self.email = os.getenv("JIRA_EMAIL")
        self.token = os.getenv("JIRA_API_TOKEN")
        self.cloud_id = os.getenv("JIRA_CLOUD_ID")
        
        if not all([self.site, self.email, self.token]):
            raise ValueError("Missing JIRA_SITE / JIRA_EMAIL / JIRA_API_TOKEN in .env")
        
        self.auth = HTTPBasicAuth(self.email, self.token)
        self.headers = {"Accept": "application/json", "Content-Type": "application/json"}
        self._base_url = None
    
    def get_cloud_id(self) -> str:
        """Get Jira Cloud ID for scoped token authentication."""
        if self.cloud_id:
            return self.cloud_id
        
        r = requests.get(f"{self.site}/_edge/tenant_info", timeout=30)
        r.raise_for_status()
        self.cloud_id = r.json()["cloudId"]
        return self.cloud_id
    
    @property
    def base_url(self) -> str:
        """Get base API URL."""
        if not self._base_url:
            self._base_url = f"https://api.atlassian.com/ex/jira/{self.get_cloud_id()}"
        return self._base_url
    
    @staticmethod
    def extract_issue_key(input_str: str) -> str:
        """Extract issue key from URL or return as-is if already a key."""
        # URL-decode the input first (handles cases where URL is encoded)
        decoded_input = unquote(input_str)
        
        # Check if it's a URL
        if "atlassian.net" in decoded_input or "jira.com" in decoded_input:
            match = re.search(r'/browse/([A-Z]+-\d+)', decoded_input)
            if match:
                return match.group(1)
        
        # Check if it's already a valid issue key format
        if re.match(r'^[A-Z]+-\d+$', decoded_input):
            return decoded_input
        
        raise ValueError(f"Invalid issue key or URL: {input_str}")
    
    @staticmethod
    def adf_to_text(node: Any) -> str:
        """Flatten Atlassian Document Format (rich text JSON) into plain text."""
        if not isinstance(node, dict):
            return ""
        if node.get("type") == "text":
            return node.get("text", "")
        joined = "".join(JiraClient.adf_to_text(child) for child in node.get("content", []))
        if node.get("type") in ("paragraph", "heading", "listItem"):
            joined += "\n"
        return joined
    
    @staticmethod
    def text_to_adf(text: str) -> Dict[str, Any]:
        """Convert plain text to Atlassian Document Format."""
        paragraphs = text.split("\n")
        content = []
        
        for para in paragraphs:
            if para.strip():
                content.append({
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "text": para
                        }
                    ]
                })
        
        return {
            "version": 1,
            "type": "doc",
            "content": content if content else [
                {
                    "type": "paragraph",
                    "content": []
                }
            ]
        }
    
    def fetch_issue(self, issue_key: str) -> Dict[str, Any]:
        """Fetch a Jira issue by key."""
        key = self.extract_issue_key(issue_key)
        
        r = requests.get(
            f"{self.base_url}/rest/api/3/issue/{key}",
            auth=self.auth,
            headers=self.headers,
            params={
                "fields": "summary,status,assignee,reporter,priority,issuetype,"
                          "created,updated,duedate,labels,description,parent,project"
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        
        # Convert description from ADF to text for easier frontend handling
        if data["fields"].get("description"):
            data["fields"]["description_text"] = self.adf_to_text(
                data["fields"]["description"]
            ).strip()
        
        return data
    
    def fetch_comments(self, issue_key: str) -> List[Dict[str, Any]]:
        """Fetch comments for a Jira issue."""
        key = self.extract_issue_key(issue_key)
        
        r = requests.get(
            f"{self.base_url}/rest/api/3/issue/{key}/comment",
            auth=self.auth,
            headers=self.headers,
            params={"maxResults": 100},
            timeout=30,
        )
        r.raise_for_status()
        comments = r.json()["comments"]
        
        # Convert comment body from ADF to text
        for comment in comments:
            if comment.get("body"):
                comment["body_text"] = self.adf_to_text(comment["body"]).strip()
        
        return comments
    
    def fetch_changelog(self, issue_key: str) -> List[Dict[str, Any]]:
        """Fetch full field-change history for a Jira issue."""
        key = self.extract_issue_key(issue_key)
        
        entries, start_at = [], 0
        while True:
            r = requests.get(
                f"{self.base_url}/rest/api/3/issue/{key}/changelog",
                auth=self.auth,
                headers=self.headers,
                params={"startAt": start_at, "maxResults": 100},
                timeout=30,
            )
            r.raise_for_status()
            page = r.json()
            entries.extend(page["values"])
            start_at += len(page["values"])
            if not page["values"] or start_at >= page.get("total", 0):
                break
        
        return entries
    
    def create_issue(
        self,
        project_key: str,
        summary: str,
        description: str,
        issue_type: str = "Task",
        priority: Optional[str] = None,
        assignee_id: Optional[str] = None,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new Jira issue."""
        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": self.text_to_adf(description),
                "issuetype": {"name": issue_type}
            }
        }
        
        if priority:
            payload["fields"]["priority"] = {"name": priority}
        
        if assignee_id:
            payload["fields"]["assignee"] = {"id": assignee_id}
        
        if labels:
            payload["fields"]["labels"] = labels
        
        r = requests.post(
            f"{self.base_url}/rest/api/3/issue",
            auth=self.auth,
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    
    def update_issue(
        self,
        issue_key: str,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        assignee_id: Optional[str] = None,
        priority: Optional[str] = None,
        labels: Optional[List[str]] = None,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an existing Jira issue."""
        key = self.extract_issue_key(issue_key)
        
        fields = {}
        
        if summary is not None:
            fields["summary"] = summary
        
        if description is not None:
            fields["description"] = self.text_to_adf(description)
        
        if assignee_id is not None:
            fields["assignee"] = {"id": assignee_id} if assignee_id else None
        
        if priority is not None:
            fields["priority"] = {"name": priority}
        
        if labels is not None:
            fields["labels"] = labels
        
        payload = {"fields": fields}
        
        r = requests.put(
            f"{self.base_url}/rest/api/3/issue/{key}",
            auth=self.auth,
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        
        # Handle status transition separately if provided
        if status:
            self._transition_issue(key, status)
        
        return {"message": "Issue updated successfully", "key": key}
    
    def _transition_issue(self, issue_key: str, status_name: str):
        """Transition an issue to a new status."""
        # Get available transitions
        r = requests.get(
            f"{self.base_url}/rest/api/3/issue/{issue_key}/transitions",
            auth=self.auth,
            headers=self.headers,
            timeout=30,
        )
        r.raise_for_status()
        transitions = r.json()["transitions"]
        
        # Find the transition ID for the target status
        transition_id = None
        for transition in transitions:
            if transition["to"]["name"].lower() == status_name.lower():
                transition_id = transition["id"]
                break
        
        if not transition_id:
            raise ValueError(f"Cannot transition to status '{status_name}'. Available: {[t['to']['name'] for t in transitions]}")
        
        # Perform the transition
        r = requests.post(
            f"{self.base_url}/rest/api/3/issue/{issue_key}/transitions",
            auth=self.auth,
            headers=self.headers,
            json={"transition": {"id": transition_id}},
            timeout=30,
        )
        r.raise_for_status()
    
    def get_issue_full_details(self, issue_key: str) -> Dict[str, Any]:
        """Get complete issue details including comments and changelog."""
        issue = self.fetch_issue(issue_key)
        comments = self.fetch_comments(issue_key)
        changelog = self.fetch_changelog(issue_key)
        
        return {
            "issue": issue,
            "comments": comments,
            "changelog": changelog
        }
