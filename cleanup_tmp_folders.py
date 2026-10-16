#!/usr/bin/env python3
"""
Standalone script to clean up accumulated /tmp/ directories created by CrewAI agent execution.
This addresses the issue where folders like /tmp/cs_1c8b03 accumulate over time and consume disk space.

Usage:
    python cleanup_tmp_folders.py --dry-run    # See what would be deleted
    python cleanup_tmp_folders.py              # Actually delete old folders
    python cleanup_tmp_folders.py --max-age 12 # Delete folders older than 12 hours
"""

import os
import sys
import glob
import shutil
import fnmatch
import time
import logging
import argparse
from datetime import datetime
from typing import Dict


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("tmp_cleanup")


def get_directory_size_mb(path: str) -> float:
    """Calculate directory size in MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, IOError):
                    pass
    except Exception:
        pass
    return total_size / (1024 * 1024)


def cleanup_old_tmp_directories(max_age_hours: int = 24, dry_run: bool = False, verbose: bool = False) -> Dict[str, int]:
    """
    Clean up old accumulated /tmp/ directories created by CrewAI execution.
    
    Args:
        max_age_hours: Delete directories older than this many hours
        dry_run: If True, only report what would be deleted without actually deleting
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with cleanup statistics
    """
    logger = setup_logging(verbose)
    current_time = time.time()
    age_threshold = max_age_hours * 3600  # Convert to seconds
    
    # Patterns for directories that can be safely deleted (these are the culprits!)
    cleanup_patterns = [
        '/tmp/cm_*',           # Memory directories (/tmp/cm_{short_id})
        '/tmp/cc_*',           # Cache directories (/tmp/cc_{short_id})
        '/tmp/cs_*',           # Storage directories (/tmp/cs_{short_id}) - THE MAIN ISSUE!
        '/tmp/crew_safe_*',    # Safe path directories
        '/tmp/crew_default*',  # Default crew storage
        '/tmp/crewai_base*',   # Base CrewAI storage
        '/tmp/crewai_memory_*', # Memory paths with full execution IDs
        '/tmp/crewai_cache_*',  # Cache paths with full execution IDs
        '/tmp/*crewai*tmp*',   # Any crewai temp files
        '/tmp/*crew*temp*',    # Any crew temp files
        '/tmp/tmp*crew*',      # Any other crew temp files
    ]
    
    # Directories to always preserve (contain valuable reusable data)
    preserve_patterns = [
        '/tmp/embed_cache*',        # Embedding cache - preserve for reuse
        '/tmp/vector_store*',       # Vector store - preserve for reuse  
        '/tmp/crewai_db*',          # Database storage - may contain important data
        '/tmp/crewai_storage_kb_*', # Knowledge base storage - preserve for reuse
    ]
    
    stats = {
        'scanned': 0,
        'deleted': 0,
        'preserved': 0,
        'errors': 0,
        'total_size_mb': 0,
        'patterns_found': {}
    }
    
    try:
        logger.info(f"🧹 Starting cleanup of /tmp/ directories older than {max_age_hours} hours")
        if dry_run:
            logger.info("🔍 DRY RUN MODE - No files will actually be deleted")
        
        # Collect all candidates
        all_candidates = set()
        for pattern in cleanup_patterns:
            matching_items = glob.glob(pattern)
            if matching_items:
                stats['patterns_found'][pattern] = len(matching_items)
                logger.info(f"📁 Found {len(matching_items)} items matching pattern: {pattern}")
            all_candidates.update(matching_items)
        
        if not all_candidates:
            logger.info("✅ No temporary directories found to clean up!")
            return stats
        
        logger.info(f"📊 Total candidates for cleanup: {len(all_candidates)}")
        
        for item_path in sorted(all_candidates):
            stats['scanned'] += 1
            
            try:
                # Check if item exists and get its age
                if not os.path.exists(item_path):
                    logger.debug(f"⚠️  Item no longer exists: {item_path}")
                    continue
                    
                # Get modification time
                mtime = os.path.getmtime(item_path)
                age_seconds = current_time - mtime
                age_hours = age_seconds / 3600
                
                # Skip if not old enough
                if age_seconds < age_threshold:
                    logger.debug(f"⏰ Skipping recent item: {os.path.basename(item_path)} (age: {age_hours:.1f}h)")
                    continue
                
                # Check if this should be preserved
                should_preserve = False
                for preserve_pattern in preserve_patterns:
                    if fnmatch.fnmatch(item_path, preserve_pattern):
                        should_preserve = True
                        stats['preserved'] += 1
                        logger.info(f"🛡️  Preserved important directory: {item_path}")
                        break
                
                if should_preserve:
                    continue
                
                # Calculate size before deletion
                size_mb = 0
                if os.path.isdir(item_path):
                    size_mb = get_directory_size_mb(item_path)
                elif os.path.isfile(item_path):
                    try:
                        size_mb = os.path.getsize(item_path) / (1024 * 1024)
                    except (OSError, IOError):
                        pass
                
                stats['total_size_mb'] += size_mb
                
                # Delete the item (unless dry run)
                action_verb = "Would delete" if dry_run else "Deleting"
                logger.info(f"🗑️  {action_verb}: {item_path} (age: {age_hours:.1f}h, size: {size_mb:.1f}MB)")
                
                if not dry_run:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                
                stats['deleted'] += 1
                
            except Exception as e:
                stats['errors'] += 1
                logger.warning(f"❌ Error processing {item_path}: {e}")
        
        # Final summary
        if stats['deleted'] > 0:
            action = "Would free" if dry_run else "Freed"
            logger.info(f"🎉 Cleanup completed successfully!")
            logger.info(f"📈 Summary: scanned={stats['scanned']}, deleted={stats['deleted']}, "
                       f"preserved={stats['preserved']}, errors={stats['errors']}")
            logger.info(f"💾 {action} {stats['total_size_mb']:.1f}MB of disk space")
        else:
            logger.info("✅ No old temporary directories found to clean up")
        
        if stats['patterns_found']:
            logger.info("📋 Patterns that matched:")
            for pattern, count in stats['patterns_found'].items():
                logger.info(f"   {pattern}: {count} items")
        
    except Exception as e:
        logger.error(f"💥 Error during /tmp/ cleanup: {e}")
        stats['errors'] += 1
    
    return stats


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Clean up accumulated CrewAI temporary directories in /tmp/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run                    # See what would be deleted
  %(prog)s                             # Actually delete old folders (24h+)
  %(prog)s --max-age 12                # Delete folders older than 12 hours
  %(prog)s --max-age 6 --verbose       # Delete 6h+ old folders with verbose output
        """
    )
    
    parser.add_argument(
        '--max-age',
        type=int,
        default=24,
        help='Delete directories older than this many hours (default: 24)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    print("🚀 CrewAI Temporary Directory Cleanup Tool")
    print("=" * 50)
    
    # Run the cleanup
    stats = cleanup_old_tmp_directories(
        max_age_hours=args.max_age,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    print("=" * 50)
    if args.dry_run:
        print(f"🔍 DRY RUN COMPLETE - Would have freed {stats['total_size_mb']:.1f}MB")
        print("   Run without --dry-run to actually delete the files")
    else:
        print(f"✅ CLEANUP COMPLETE - Freed {stats['total_size_mb']:.1f}MB of disk space")
    
    return 0 if stats['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
