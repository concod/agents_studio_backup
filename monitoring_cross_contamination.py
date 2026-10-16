#!/usr/bin/env python3
"""
Cross-Contamination Monitor for Agent Execution System

This script monitors log files for signs of cross-contamination between concurrent requests
and provides alerts when such issues are detected.

Usage:
    python monitoring_cross_contamination.py --log-dir logs --monitor
"""

import os
import re
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set
from pathlib import Path
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class CrossContaminationMonitor:
    def __init__(self, log_dir: str, alert_email: str = None, smtp_config: dict = None):
        self.log_dir = Path(log_dir)
        self.alert_email = alert_email
        self.smtp_config = smtp_config or {}
        self.processed_files: Set[str] = set()
        self.contamination_alerts: List[Dict] = []
        
    def scan_log_files(self) -> List[Path]:
        """Scan for agent execution log files"""
        pattern = "agent_execution_*.log"
        return list(self.log_dir.glob(pattern))
    
    def analyze_log_file(self, log_file: Path) -> Dict:
        """Analyze a single log file for cross-contamination signs"""
        analysis = {
            'file': str(log_file),
            'execution_id': None,
            'user_inputs': [],
            'contamination_detected': False,
            'issues': []
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract execution ID from filename
            match = re.search(r'agent_execution_([^.]+)\.log', log_file.name)
            if match:
                analysis['execution_id'] = match.group(1)
            
            # Find all user inputs in the log
            input_patterns = [
                r'userInput=([^,\n]+)',
                r'EXECUTION CONTEXT SET - Input: \'([^\']+)\'',
                r'Tool processing input: \'([^\']+)\'',
                r'Generating payload and endpoint URL for input: ([^\n]+)'
            ]
            
            all_inputs = []
            for pattern in input_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                all_inputs.extend(matches)
            
            # Clean and deduplicate inputs
            unique_inputs = []
            for inp in all_inputs:
                cleaned = inp.strip().strip("'\"")
                if cleaned and cleaned not in unique_inputs:
                    unique_inputs.append(cleaned)
            
            analysis['user_inputs'] = unique_inputs
            
            # Check for contamination signs - be more intelligent about what constitutes contamination
            if len(unique_inputs) > 1:
                # Check if inputs are actually different enough to be contamination
                # Allow for minor variations and processing
                significantly_different = False
                for i, input1 in enumerate(unique_inputs):
                    for j, input2 in enumerate(unique_inputs[i+1:], i+1):
                        words1 = set(input1.lower().split())
                        words2 = set(input2.lower().split())
                        
                        # If less than 30% word overlap and both are substantial, it's likely contamination
                        overlap = len(words1.intersection(words2)) / max(len(words1.union(words2)), 1)
                        if overlap < 0.3 and len(input1) > 20 and len(input2) > 20:
                            significantly_different = True
                            break
                    if significantly_different:
                        break
                
                if significantly_different:
                    analysis['contamination_detected'] = True
                    analysis['issues'].append(f"Significantly different inputs detected (possible contamination): {unique_inputs}")
                else:
                    # Just log as a warning, not contamination
                    analysis['issues'].append(f"Input variations detected (likely normal processing): {unique_inputs}")
            
            # Look for explicit contamination alerts
            contamination_alerts = re.findall(r'CRITICAL ALERT: Cross-contamination detected.*', content)
            if contamination_alerts:
                analysis['contamination_detected'] = True
                analysis['issues'].extend(contamination_alerts)
            
            # Look for critical input mismatch errors (not warnings)
            mismatch_errors = re.findall(r'INPUT MISMATCH DETECTED.*', content)
            if mismatch_errors:
                analysis['contamination_detected'] = True
                analysis['issues'].extend(mismatch_errors)
                
        except Exception as e:
            logger.error(f"Error analyzing {log_file}: {e}")
            analysis['issues'].append(f"Analysis error: {e}")
            
        return analysis
    
    def send_alert(self, contamination_data: Dict):
        """Send alert about contamination detection"""
        if not self.alert_email or not self.smtp_config:
            logger.warning("Email alerting not configured")
            return
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config.get('from_email', 'monitor@yourdomain.com')
            msg['To'] = self.alert_email
            msg['Subject'] = f"CRITICAL: Cross-Contamination Detected - {contamination_data['execution_id']}"
            
            body = f"""
CRITICAL ALERT: Cross-contamination detected in agent execution system

Execution ID: {contamination_data['execution_id']}
Log File: {contamination_data['file']}
Timestamp: {datetime.now()}

User Inputs Found:
{chr(10).join(f"- {inp}" for inp in contamination_data['user_inputs'])}

Issues Detected:
{chr(10).join(f"- {issue}" for issue in contamination_data['issues'])}

This indicates that concurrent requests may be getting mixed up, leading to users receiving 
responses to other users' queries. This is a critical security and functionality issue that 
requires immediate attention.

Please investigate the CrewAI execution isolation and concurrent request handling immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_config.get('host', 'localhost'), 
                                self.smtp_config.get('port', 587))
            if self.smtp_config.get('use_tls'):
                server.starttls()
            if self.smtp_config.get('username'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            server.sendmail(msg['From'], [self.alert_email], msg.as_string())
            server.quit()
            
            logger.info(f"Alert sent for contamination in {contamination_data['execution_id']}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def monitor_continuously(self, check_interval: int = 30):
        """Continuously monitor log files for contamination"""
        logger.info(f"Starting continuous monitoring of {self.log_dir}")
        logger.info(f"Check interval: {check_interval} seconds")
        
        while True:
            try:
                log_files = self.scan_log_files()
                new_files = [f for f in log_files if str(f) not in self.processed_files]
                
                if new_files:
                    logger.info(f"Found {len(new_files)} new log files to analyze")
                
                for log_file in new_files:
                    analysis = self.analyze_log_file(log_file)
                    
                    if analysis['contamination_detected']:
                        logger.critical(f"Cross-contamination detected in {log_file.name}")
                        self.contamination_alerts.append(analysis)
                        self.send_alert(analysis)
                    else:
                        logger.debug(f"No contamination detected in {log_file.name}")
                    
                    self.processed_files.add(str(log_file))
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during monitoring: {e}")
                time.sleep(check_interval)
    
    def analyze_historical_logs(self) -> Dict:
        """Analyze all historical log files"""
        logger.info("Analyzing historical log files...")
        
        log_files = self.scan_log_files()
        results = {
            'total_files': len(log_files),
            'contaminated_files': 0,
            'contamination_details': []
        }
        
        for log_file in log_files:
            analysis = self.analyze_log_file(log_file)
            
            if analysis['contamination_detected']:
                results['contaminated_files'] += 1
                results['contamination_details'].append(analysis)
                logger.warning(f"Contamination found in {log_file.name}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Monitor for cross-contamination in agent logs')
    parser.add_argument('--log-dir', default='logs', help='Directory containing log files')
    parser.add_argument('--monitor', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--analyze-historical', action='store_true', help='Analyze historical logs')
    parser.add_argument('--alert-email', help='Email address for alerts')
    parser.add_argument('--check-interval', type=int, default=30, help='Monitoring check interval in seconds')
    
    args = parser.parse_args()
    
    # SMTP configuration (you can modify this based on your email provider)
    smtp_config = {
        'host': 'smtp.gmail.com',  # Change to your SMTP server
        'port': 587,
        'use_tls': True,
        'username': os.getenv('SMTP_USERNAME'),  # Set these environment variables
        'password': os.getenv('SMTP_PASSWORD'),
        'from_email': os.getenv('SMTP_FROM_EMAIL', 'monitor@yourdomain.com')
    }
    
    monitor = CrossContaminationMonitor(
        log_dir=args.log_dir,
        alert_email=args.alert_email,
        smtp_config=smtp_config
    )
    
    if args.analyze_historical:
        results = monitor.analyze_historical_logs()
        print(f"\nHistorical Analysis Results:")
        print(f"Total files analyzed: {results['total_files']}")
        print(f"Files with contamination: {results['contaminated_files']}")
        
        if results['contaminated_files'] > 0:
            print(f"\nContamination Details:")
            for detail in results['contamination_details']:
                print(f"- {detail['execution_id']}: {len(detail['user_inputs'])} different inputs")
                for issue in detail['issues']:
                    print(f"  * {issue}")
        
        # Save detailed results
        with open('contamination_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to contamination_analysis.json")
    
    if args.monitor:
        monitor.monitor_continuously(args.check_interval)

if __name__ == '__main__':
    main() 