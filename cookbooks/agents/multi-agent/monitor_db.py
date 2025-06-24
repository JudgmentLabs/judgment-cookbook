"""
ChromaDB State Monitor
Continuously monitors the state of ChromaDB.
"""

import time
import json
import os
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
import chromadb
from collections import defaultdict, deque
import threading
import argparse
import signal
import sys
from queue import Queue, Empty

class ChromaDBMonitor:
    def __init__(self, db_path: str = "./memory_db", collection_name: str = "documents", 
                 refresh_interval: int = 5, detailed_logging: bool = False, 
                 realtime_monitoring: bool = True, simple_mode: bool = False):
        """
        Initialize ChromaDB monitor.
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection to monitor
            refresh_interval: Seconds between status updates
            detailed_logging: Whether to show detailed document information
            realtime_monitoring: Whether to monitor operations in real-time
            simple_mode: If True, only show count changes without document details
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.refresh_interval = refresh_interval
        self.detailed_logging = detailed_logging
        self.realtime_monitoring = realtime_monitoring
        self.simple_mode = simple_mode
        self.running = True
        
        # Performance tracking
        self.query_times = deque(maxlen=100)  # Keep last 100 query times
        self.add_times = deque(maxlen=100)    # Keep last 100 add times
        self.error_count = 0
        self.last_error = None
        
        # Real-time monitoring
        self.last_document_count = 0
        self.last_check_time = datetime.now()
        self.operations_log = deque(maxlen=100)
        self.operation_queue = Queue()

        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            print(f"✅ Connected to ChromaDB at: {os.path.abspath(self.db_path)}")
        except Exception as e:
            print(f"❌ Failed to connect to ChromaDB: {e}")
            sys.exit(1)
            
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self._initialize_baseline()
    
    def _initialize_baseline(self):
        """Initialize baseline metrics for comparison."""
        try:
            collection = self.client.get_or_create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.last_document_count = collection.count()
            self.last_check_time = datetime.now()
            print(f"📊 Baseline: {self.last_document_count} documents")
        except Exception as e:
            print(f"⚠️ Could not establish baseline: {e}")
            self.last_document_count = 0
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.running = False
    
    def log_operation(self, operation_type: str, details: str, metadata: Dict = None):
        """Log a database operation."""
        timestamp = datetime.now()
        operation = {
            'timestamp': timestamp,
            'type': operation_type,
            'details': details,
            'metadata': metadata or {}
        }
        self.operations_log.append(operation)
        self.operation_queue.put(operation)
    
    def monitor_realtime_operations(self):
        """Monitor database for real-time changes using timestamps."""
        print("🔄 Starting real-time operation monitoring...")
        
        if self.simple_mode:
            print("📊 Simple mode enabled - tracking count changes only")
        else:
            print("⏰ Using timestamp-based new document detection")
        
        last_check_timestamp = datetime.now()
        
        while self.running:
            try:
                current_time = datetime.now()
                
                collection = self.client.get_or_create_collection(
                    self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                current_count = collection.count()
                
                if current_count != self.last_document_count:
                    count_diff = current_count - self.last_document_count
                    
                    if count_diff > 0:

                        self.log_operation(
                            "INSERT", 
                            f"{count_diff} document(s) added (Total: {current_count:,})",
                            {"count_change": count_diff, "total_count": current_count}
                        )
                        
                        if not self.simple_mode:
                            try:
                                all_docs = collection.get(include=['metadatas', 'documents'])
                                new_documents_found = []
                                
                                if all_docs['metadatas'] and all_docs['documents']:
                                    for doc, meta in zip(all_docs['documents'], all_docs['metadatas']):
                                        if meta and 'timestamp' in meta:
                                            try:
                                                doc_timestamp_str = meta['timestamp']
                                                # Handle both formats: with and without 'Z' suffix
                                                if doc_timestamp_str.endswith('Z'):
                                                    doc_timestamp = datetime.fromisoformat(doc_timestamp_str.replace('Z', '+00:00'))
                                                else:
                                                    doc_timestamp = datetime.fromisoformat(doc_timestamp_str)
                                                
                                                if doc_timestamp > last_check_timestamp:
                                                    new_documents_found.append((doc, meta, doc_timestamp))
                                            except (ValueError, TypeError) as e:
                                                continue
                                
                                # Sort by timestamp (newest first) and log details
                                if new_documents_found:
                                    new_documents_found.sort(key=lambda x: x[2], reverse=True)
                                    
                                    for doc, meta, doc_timestamp in new_documents_found[:5]:
                                        time_str = doc_timestamp.strftime("%H:%M:%S")
                                        self.log_operation(
                                            "INSERT_DETAIL",
                                            f"New document ({time_str}): ID={meta.get('memory_id', 'unknown')}, Size={len(doc)} chars",
                                            {
                                                "content_preview": doc[:100] + "..." if len(doc) > 100 else doc,
                                                "tags": meta.get('tags', ''),
                                                "importance": meta.get('importance', 'unknown'),
                                                "memory_id": meta.get('memory_id', 'unknown'),
                                                "timestamp": meta.get('timestamp', 'unknown'),
                                                "parsed_timestamp": doc_timestamp.isoformat()
                                            }
                                        )
                                    
                                    if len(new_documents_found) > 5:
                                        self.log_operation(
                                            "INSERT_DETAIL",
                                            f"... and {len(new_documents_found) - 5} more documents",
                                            {"additional_count": len(new_documents_found) - 5}
                                        )
                                
                                elif count_diff > 0:
                                    self.log_operation(
                                        "INSERT_DETAIL",
                                        f"Document count increased but no new timestamps detected (documents may lack timestamps)",
                                        {"note": "Some documents may not have proper timestamp metadata"}
                                    )
                            
                            except Exception as e:
                                self.log_operation("ERROR", f"Could not analyze new documents by timestamp: {e}")
                    
                    elif count_diff < 0:
                        self.log_operation(
                            "DELETE", 
                            f"{abs(count_diff)} document(s) removed (Total: {current_count:,})",
                            {"count_change": count_diff, "total_count": current_count}
                        )
                    
                    self.last_document_count = current_count
                
                last_check_timestamp = current_time
                self.last_check_time = current_time
                
                time.sleep(1)
                
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                self.log_operation("ERROR", f"Monitoring error: {e}")
                time.sleep(5)
                
    def print_operations_feed(self):
        """Print real-time operations as they happen."""
        while self.running:
            try:
                # Get operation from queue (blocks until available or timeout)
                operation = self.operation_queue.get(timeout=1)
                
                timestamp = operation['timestamp'].strftime("%H:%M:%S")
                op_type = operation['type']
                details = operation['details']
                
                if op_type == "INSERT":
                    icon = "📝"
                    color_code = "\033[92m"  # Green
                elif op_type == "INSERT_DETAIL":
                    icon = "  ↳"
                    color_code = "\033[94m"  # Blue
                elif op_type == "DELETE":
                    icon = "🗑️"
                    color_code = "\033[91m"  # Red
                elif op_type == "QUERY":
                    icon = "🔍"
                    color_code = "\033[93m"  # Yellow
                elif op_type == "ERROR":
                    icon = "❌"
                    color_code = "\033[91m"  # Red
                else:
                    icon = "ℹ️"
                    color_code = "\033[0m"   # Default
                
                reset_code = "\033[0m"
                
                print(f"{color_code}[{timestamp}] {icon} {details}{reset_code}")
                
                if self.detailed_logging and operation.get('metadata'):
                    metadata = operation['metadata']
                    if 'content_preview' in metadata:
                        print(f"        Content: {metadata['content_preview']}")
                    if 'tags' in metadata and metadata['tags']:
                        print(f"        Tags: {metadata['tags']}")
                    if 'importance' in metadata and metadata['importance'] != 'unknown':
                        print(f"        Importance: {metadata['importance']}")
                
            except Empty:
                continue
            except Exception as e:
                print(f"❌ Error in operations feed: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get detailed information about the collection."""
        try:
            start_time = time.time()
            
            collection = self.client.get_or_create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            count = collection.count()
            
            limit = min(count, 1000) if count > 0 else 0
            result = collection.get(limit=limit, include=['metadatas', 'documents'])
            
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            
            if self.realtime_monitoring:
                self.log_operation(
                    "QUERY", 
                    f"Status query: {count:,} documents, {query_time:.3f}s",
                    {"query_time_ms": round(query_time * 1000, 2), "document_count": count}
                )
            
            return {
                'collection_name': self.collection_name,
                'total_documents': count,
                'retrieved_documents': len(result['documents']) if result['documents'] else 0,
                'metadatas': result['metadatas'] if result['metadatas'] else [],
                'documents': result['documents'] if result['documents'] else [],
                'query_time': query_time,
                'success': True
            }
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            if self.realtime_monitoring:
                self.log_operation("ERROR", f"Query failed: {e}")
            return {
                'collection_name': self.collection_name,
                'error': str(e),
                'success': False
            }
    
    def analyze_metadata(self, metadatas: List[Dict]) -> Dict[str, Any]:
        """Analyze metadata to extract insights."""
        if not metadatas:
            return {}
        
        importance_counts = defaultdict(int)
        tag_counts = defaultdict(int)
        agent_counts = defaultdict(int)
        
        recent_docs = 0
        old_docs = 0
        current_time = datetime.now()
        
        for metadata in metadatas:
            importance = metadata.get('importance', 'unknown')
            importance_counts[importance] += 1
            
            tags = metadata.get('tags', '')
            if tags:
                for tag in tags.split(','):
                    tag = tag.strip()
                    if tag:
                        tag_counts[tag] += 1
            
            timestamp_str = metadata.get('timestamp', '')
            if timestamp_str:
                try:
                    doc_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if current_time - doc_time < timedelta(hours=24):
                        recent_docs += 1
                    else:
                        old_docs += 1
                except:
                    pass
        
        return {
            'importance_distribution': dict(importance_counts),
            'tag_distribution': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'recent_documents_24h': recent_docs,
            'older_documents': old_docs
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system resource usage statistics."""
        try:
            db_size = 0
            if os.path.exists(self.db_path):
                for dirpath, dirnames, filenames in os.walk(self.db_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            db_size += os.path.getsize(filepath)
                        except OSError:
                            pass
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            disk_usage = psutil.disk_usage(os.path.dirname(os.path.abspath(self.db_path)))
            
            return {
                'database_size_mb': round(db_size / (1024 * 1024), 2),
                'memory_usage_mb': round(memory_info.rss / (1024 * 1024), 2),
                'disk_free_gb': round(disk_usage.free / (1024 * 1024 * 1024), 2),
                'disk_total_gb': round(disk_usage.total / (1024 * 1024 * 1024), 2),
                'cpu_percent': psutil.cpu_percent()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        if self.query_times:
            query_times_list = list(self.query_times)
            stats['avg_query_time_ms'] = round(sum(query_times_list) / len(query_times_list) * 1000, 2)
            stats['max_query_time_ms'] = round(max(query_times_list) * 1000, 2)
            stats['min_query_time_ms'] = round(min(query_times_list) * 1000, 2)
        
        if self.add_times:
            add_times_list = list(self.add_times)
            stats['avg_add_time_ms'] = round(sum(add_times_list) / len(add_times_list) * 1000, 2)
            stats['max_add_time_ms'] = round(max(add_times_list) * 1000, 2)
            stats['min_add_time_ms'] = round(min(add_times_list) * 1000, 2)
        
        stats['total_errors'] = self.error_count
        stats['last_error'] = self.last_error
        stats['total_operations'] = len(self.operations_log)
        
        recent_ops = 0
        cutoff_time = datetime.now() - timedelta(minutes=5)
        for op in self.operations_log:
            if op['timestamp'] > cutoff_time:
                recent_ops += 1
        stats['recent_operations_5min'] = recent_ops
        
        return stats
    
    def print_status_header(self, collection_info: Dict, metadata_analysis: Dict, 
                           system_stats: Dict, performance_stats: Dict):
        """Print status header (called periodically)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "=" * 80)
        print(f"📊 STATUS UPDATE - {timestamp}")
        print("=" * 80)
        
        if not collection_info.get('success', False):
            print(f"❌ ERROR: {collection_info.get('error', 'Unknown error')}")
            return
        
        print(f"📊 COLLECTION: {collection_info['collection_name']}")
        print(f"   Total Documents: {collection_info['total_documents']:,}")
        print(f"   Query Time: {collection_info['query_time']:.3f}s")
        
        if 'error' not in system_stats:
            print(f"💾 RESOURCES: DB={system_stats['database_size_mb']}MB | "
                  f"RAM={system_stats['memory_usage_mb']}MB | "
                  f"CPU={system_stats['cpu_percent']}%")
        
        recent_ops = performance_stats.get('recent_operations_5min', 0)
        total_ops = performance_stats.get('total_operations', 0)
        errors = performance_stats.get('total_errors', 0)
        
        print(f"⚡ ACTIVITY: {recent_ops} ops (5min) | {total_ops} total | {errors} errors")
        
        if metadata_analysis:
            importance_dist = metadata_analysis.get('importance_distribution', {})
            recent = metadata_analysis.get('recent_documents_24h', 0)
            
            importance_summary = " | ".join([f"{k}:{v}" for k, v in importance_dist.items()])
            print(f"📈 DATA: Recent(24h):{recent} | Importance: {importance_summary}")
        
        print("=" * 80)
        print("🔄 LIVE OPERATIONS FEED (Press Ctrl+C to stop):")
        print("-" * 80)
    
    def export_status(self, filename: str = None):
        """Export current status to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chromadb_status_{timestamp}.json"
        
        collection_info = self.get_collection_info()
        metadata_analysis = self.analyze_metadata(collection_info.get('metadatas', []))
        system_stats = self.get_system_stats()
        performance_stats = self.get_performance_stats()
        
        operations_export = []
        for op in list(self.operations_log):
            operations_export.append({
                'timestamp': op['timestamp'].isoformat(),
                'type': op['type'],
                'details': op['details'],
                'metadata': op.get('metadata', {})
            })
        
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'collection_info': collection_info,
            'metadata_analysis': metadata_analysis,
            'system_stats': system_stats,
            'performance_stats': performance_stats,
            'recent_operations': operations_export
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(status_data, f, indent=2, default=str)
            print(f"✅ Status exported to: {filename}")
        except Exception as e:
            print(f"❌ Failed to export status: {e}")
    
    def run(self):
        """Main monitoring loop with real-time operations."""
        print(f"🚀 Starting ChromaDB monitor...")
        print(f"   Database Path: {os.path.abspath(self.db_path)}")
        print(f"   Collection: {self.collection_name}")
        print(f"   Status Refresh: {self.refresh_interval}s")
        print(f"   Real-time Monitoring: {self.realtime_monitoring}")
        print(f"   Simple Mode: {self.simple_mode}")
        print(f"   Detailed Logging: {self.detailed_logging}")
        
        if self.realtime_monitoring:
            monitor_thread = threading.Thread(target=self.monitor_realtime_operations, daemon=True)
            monitor_thread.start()
      
            feed_thread = threading.Thread(target=self.print_operations_feed, daemon=True)
            feed_thread.start()
            
            collection_info = self.get_collection_info()
            metadata_analysis = self.analyze_metadata(collection_info.get('metadatas', []))
            system_stats = self.get_system_stats()
            performance_stats = self.get_performance_stats()
            
            self.print_status_header(collection_info, metadata_analysis, system_stats, performance_stats)
            
            last_status_time = time.time()
            
            try:
                while self.running:
                    current_time = time.time()
                    
                    if current_time - last_status_time >= self.refresh_interval:
                        collection_info = self.get_collection_info()
                        metadata_analysis = self.analyze_metadata(collection_info.get('metadatas', []))
                        system_stats = self.get_system_stats()
                        performance_stats = self.get_performance_stats()
                        
                        self.print_status_header(collection_info, metadata_analysis, system_stats, performance_stats)
                        last_status_time = current_time
                    
                    time.sleep(1) 
                    
            except KeyboardInterrupt:
                pass
        
        else:

            while self.running:
                try:
                    collection_info = self.get_collection_info()
                    metadata_analysis = self.analyze_metadata(collection_info.get('metadatas', []))
                    system_stats = self.get_system_stats()
                    performance_stats = self.get_performance_stats()
                    
                    self.print_status_header(collection_info, metadata_analysis, system_stats, performance_stats)
                    
                    time.sleep(self.refresh_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Monitor error: {e}")
                    time.sleep(self.refresh_interval)
        
        print("\n👋 ChromaDB monitor stopped.")

def main():
    parser = argparse.ArgumentParser(description='ChromaDB State Monitor')
    parser.add_argument('--db-path', default='./memory_db', 
                        help='Path to ChromaDB database (default: ./memory_db)')
    parser.add_argument('--collection', default='documents', 
                        help='Collection name to monitor (default: documents')
    parser.add_argument('--interval', type=int, default=30, 
                        help='Status refresh interval in seconds (default: 30)')
    parser.add_argument('--detailed', action='store_true', 
                        help='Show detailed document information')
    parser.add_argument('--no-realtime', action='store_true',
                        help='Disable real-time operation monitoring')
    parser.add_argument('--simple', action='store_true',
                        help='Simple mode - only show count changes, not document details')
    parser.add_argument('--export', type=str, metavar='FILENAME',
                        help='Export current status to JSON file and exit')
    
    args = parser.parse_args()
    
    monitor = ChromaDBMonitor(
        db_path=args.db_path,
        collection_name=args.collection,
        refresh_interval=args.interval,
        detailed_logging=args.detailed,
        realtime_monitoring=not args.no_realtime,
        simple_mode=args.simple
    )
    
    if args.export:
        monitor.export_status(args.export)
    else:
        monitor.run()

if __name__ == "__main__":
    main() 