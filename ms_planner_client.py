"""
Microsoft Planner API Client - For FastAPI integration
"""
import os
from typing import Optional, Dict, Any, List
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()


class MSPlannerClient:
    def __init__(self):
        self.tenant_id = os.getenv("MS_TENANT_ID")
        self.client_id = os.getenv("MS_CLIENT_ID")
        self.client_secret = os.getenv("MS_CLIENT_SECRET")
        self.access_token = os.getenv("MS_ACCESS_TOKEN")
        
        # For now, we'll use personal access token
        # In production, implement OAuth2 flow
        if not self.access_token:
            raise ValueError("Missing MS_ACCESS_TOKEN in .env")
        
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        self.graph_url = "https://graph.microsoft.com/v1.0"
    
    def get_my_tasks(self) -> List[Dict[str, Any]]:
        """Fetch all tasks assigned to the current user."""
        try:
            # Get all plans the user has access to
            response = requests.get(
                f"{self.graph_url}/me/planner/tasks",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("value", [])
        except Exception as e:
            print(f"Error fetching tasks: {e}")
            return []
    
    def get_task_details(self, task_id: str) -> Dict[str, Any]:
        """Fetch detailed information for a specific task."""
        try:
            response = requests.get(
                f"{self.graph_url}/planner/tasks/{task_id}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            task = response.json()
            
            # Get task details (description, checklist, etc.)
            details_response = requests.get(
                f"{self.graph_url}/planner/tasks/{task_id}/details",
                headers=self.headers,
                timeout=30
            )
            if details_response.status_code == 200:
                task["details"] = details_response.json()
            
            return task
        except Exception as e:
            raise Exception(f"Failed to fetch task details: {str(e)}")
    
    def get_plan_tasks(self, plan_id: str) -> List[Dict[str, Any]]:
        """Fetch all tasks for a specific plan."""
        try:
            response = requests.get(
                f"{self.graph_url}/planner/plans/{plan_id}/tasks",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("value", [])
        except Exception as e:
            raise Exception(f"Failed to fetch plan tasks: {str(e)}")
    
    def get_my_plans(self) -> List[Dict[str, Any]]:
        """Fetch all plans the user has access to."""
        try:
            # First get groups the user is member of
            response = requests.get(
                f"{self.graph_url}/me/memberOf",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            groups = response.json().get("value", [])
            
            all_plans = []
            for group in groups:
                if group.get("@odata.type") == "#microsoft.graph.group":
                    try:
                        plans_response = requests.get(
                            f"{self.graph_url}/groups/{group['id']}/planner/plans",
                            headers=self.headers,
                            timeout=30
                        )
                        if plans_response.status_code == 200:
                            plans = plans_response.json().get("value", [])
                            all_plans.extend(plans)
                    except:
                        continue
            
            return all_plans
        except Exception as e:
            print(f"Error fetching plans: {e}")
            return []
    
    def get_my_groups(self) -> List[Dict[str, Any]]:
        """Fetch all groups the user is a member of."""
        try:
            response = requests.get(
                f"{self.graph_url}/me/memberOf",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            groups = response.json().get("value", [])
            
            # Filter to only return actual groups (not other directory objects)
            return [g for g in groups if g.get("@odata.type") == "#microsoft.graph.group"]
        except Exception as e:
            raise Exception(f"Failed to fetch groups: {str(e)}")
    
    def create_plan(
        self,
        group_id: str,
        title: str
    ) -> Dict[str, Any]:
        """Create a new plan in a group."""
        try:
            payload = {
                "owner": group_id,
                "title": title
            }
            
            response = requests.post(
                f"{self.graph_url}/planner/plans",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to create plan: {str(e)}")
    
    def create_bucket(
        self,
        plan_id: str,
        name: str
    ) -> Dict[str, Any]:
        """Create a new bucket in a plan."""
        try:
            payload = {
                "planId": plan_id,
                "name": name
            }
            
            response = requests.post(
                f"{self.graph_url}/planner/buckets",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to create bucket: {str(e)}")
    
    def create_task(
        self,
        plan_id: str,
        bucket_id: str,
        title: str,
        assignments: Optional[Dict[str, Any]] = None,
        due_date: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new task in MS Planner."""
        try:
            payload = {
                "planId": plan_id,
                "bucketId": bucket_id,
                "title": title
            }
            
            if assignments:
                payload["assignments"] = assignments
            
            if due_date:
                payload["dueDateTime"] = due_date
            
            response = requests.post(
                f"{self.graph_url}/planner/tasks",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            task = response.json()
            
            # Add description if provided
            if description and task.get("id"):
                self.update_task_details(task["id"], description)
            
            return task
        except Exception as e:
            raise Exception(f"Failed to create task: {str(e)}")
    
    def update_task(
        self,
        task_id: str,
        etag: str,
        title: Optional[str] = None,
        percent_complete: Optional[int] = None,
        due_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an existing task."""
        try:
            payload = {}
            
            if title is not None:
                payload["title"] = title
            
            if percent_complete is not None:
                payload["percentComplete"] = percent_complete
            
            if due_date is not None:
                payload["dueDateTime"] = due_date
            
            headers = {**self.headers, "If-Match": etag}
            
            response = requests.patch(
                f"{self.graph_url}/planner/tasks/{task_id}",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to update task: {str(e)}")
    
    def update_task_details(
        self,
        task_id: str,
        description: Optional[str] = None,
        etag: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update task details (description, checklist, etc.)."""
        try:
            # Get current details to get etag if not provided
            if not etag:
                details_response = requests.get(
                    f"{self.graph_url}/planner/tasks/{task_id}/details",
                    headers=self.headers,
                    timeout=30
                )
                if details_response.status_code == 200:
                    etag = details_response.headers.get("ETag")
            
            payload = {}
            if description is not None:
                payload["description"] = description
            
            headers = {**self.headers, "If-Match": etag}
            
            response = requests.patch(
                f"{self.graph_url}/planner/tasks/{task_id}/details",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to update task details: {str(e)}")
    
    def get_plan_buckets(self, plan_id: str) -> List[Dict[str, Any]]:
        """Fetch all buckets (columns) for a plan."""
        try:
            response = requests.get(
                f"{self.graph_url}/planner/plans/{plan_id}/buckets",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("value", [])
        except Exception as e:
            raise Exception(f"Failed to fetch buckets: {str(e)}")
