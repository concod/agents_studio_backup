import os
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MailgunService:
    def __init__(self):
        self.api_key = os.getenv("MAILGUN_API_KEY")
        self.domain = os.getenv("MAILGUN_DOMAIN")
        self.from_email = os.getenv("MAILGUN_FROM_EMAIL", f"noreply@{self.domain}")
        self.base_url = f"https://api.mailgun.net/v3/{self.domain}"
        
        if not self.api_key or not self.domain:
            logger.warning("Mailgun configuration missing. Set MAILGUN_API_KEY and MAILGUN_DOMAIN environment variables.")

    def send_invitation_email(self, recipient_email: str, recipient_name: str, invitation_token: str, invited_by: str) -> bool:
        """
        Send invitation email to a new user
        
        Args:
            recipient_email: Email address of the invitee
            recipient_name: Name of the invitee
            invitation_token: Unique invitation token
            invited_by: Email of the person who sent the invitation
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        if not self.api_key or not self.domain:
            logger.error("Mailgun not configured. Cannot send invitation email.")
            return False
            
        # Get the frontend URL from environment or use default
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8000")
        invitation_url = f"{frontend_url}/accept-invitation?token={invitation_token}"
        
        subject = "Invitation to Join Agentic Studio"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .button {{ 
                    display: inline-block; 
                    padding: 12px 24px; 
                    background-color: #3498db; 
                    color: white; 
                    text-decoration: none; 
                    border-radius: 5px; 
                    margin-top: 20px;
                }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Agentic Studio</h1>
                </div>
                <div class="content">
                    <h2>Hi {recipient_name},</h2>
                    <p>You've been invited by <strong>{invited_by}</strong> to join Agentic Studio!</p>
                    <p>Agentic Studio is a powerful platform for creating and managing AI agents. Click the button below to accept your invitation and get started.</p>
                    <p><a href="{invitation_url}" class="button">Accept Invitation</a></p>
                    <p>If the button doesn't work, copy and paste this link into your browser:</p>
                    <p><code>{invitation_url}</code></p>
                    <p>This invitation will expire in 7 days. If you have any questions, please contact the person who invited you.</p>
                </div>
                <div class="footer">
                    <p>© 2024 Agentic Studio. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Welcome to Agentic Studio!
        
        Hi {recipient_name},
        
        You've been invited by {invited_by} to join Agentic Studio!
        
        Agentic Studio is a powerful platform for creating and managing AI agents. 
        
        To accept your invitation and get started, visit this link:
        {invitation_url}
        
        This invitation will expire in 7 days. If you have any questions, please contact the person who invited you.
        
        © 2024 Agentic Studio. All rights reserved.
        """
        
        data = {
            "from": f"Agentic Studio <{self.from_email}>",
            "to": [recipient_email],
            "subject": subject,
            "text": text_content,
            "html": html_content
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/messages",
                auth=("api", self.api_key),
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Invitation email sent successfully to {recipient_email}")
                return True
            else:
                logger.error(f"Failed to send invitation email. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending invitation email: {str(e)}")
            return False

    def send_welcome_email(self, recipient_email: str, recipient_name: str) -> bool:
        """
        Send welcome email to a user who accepted their invitation
        
        Args:
            recipient_email: Email address of the user
            recipient_name: Name of the user
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        if not self.api_key or not self.domain:
            logger.error("Mailgun not configured. Cannot send welcome email.")
            return False
            
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8000")
        
        subject = "Welcome to Agentic Studio!"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #27ae60; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .button {{ 
                    display: inline-block; 
                    padding: 12px 24px; 
                    background-color: #3498db; 
                    color: white; 
                    text-decoration: none; 
                    border-radius: 5px; 
                    margin-top: 20px;
                }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Agentic Studio!</h1>
                </div>
                <div class="content">
                    <h2>Hi {recipient_name},</h2>
                    <p>Congratulations! You've successfully joined Agentic Studio.</p>
                    <p>You can now access all the features of our AI agent platform:</p>
                    <ul>
                        <li>Create and manage AI agents</li>
                        <li>Design custom workflows</li>
                        <li>Access the agent marketplace</li>
                        <li>Collaborate with your team</li>
                    </ul>
                    <p><a href="{frontend_url}" class="button">Start Building</a></p>
                    <p>If you have any questions or need help getting started, don't hesitate to reach out to our support team.</p>
                </div>
                <div class="footer">
                    <p>© 2024 Agentic Studio. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Welcome to Agentic Studio!
        
        Hi {recipient_name},
        
        Congratulations! You've successfully joined Agentic Studio.
        
        You can now access all the features of our AI agent platform:
        - Create and manage AI agents
        - Design custom workflows  
        - Access the agent marketplace
        - Collaborate with your team
        
        Visit {frontend_url} to start building.
        
        If you have any questions or need help getting started, don't hesitate to reach out to our support team.
        
        © 2024 Agentic Studio. All rights reserved.
        """
        
        data = {
            "from": f"Agentic Studio <{self.from_email}>",
            "to": [recipient_email],
            "subject": subject,
            "text": text_content,
            "html": html_content
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/messages",
                auth=("api", self.api_key),
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Welcome email sent successfully to {recipient_email}")
                return True
            else:
                logger.error(f"Failed to send welcome email. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending welcome email: {str(e)}")
            return False 