#!/usr/bin/env python3
"""
AI File Processor - Automated file analysis using GenAI models
Supports both Ollama (local) and OpenAI API for processing text files and images
"""

import os
import json
import subprocess
import requests
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import base64
import mimetypes
import threading
import queue

# Third-party imports
try:
    from dotenv import load_dotenv
    import openai
    from PIL import Image
    import magic  # python-magic for file type detection
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install python-dotenv openai pillow python-magic")
    exit(1)

# Load environment variables
load_dotenv()

# Configuration
@dataclass
class Config:
    """Configuration settings for the file processor"""
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2-vision"  # Model that supports vision
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    use_ollama: bool = True
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    batch_size: int = 5
    priority_folders: List[str] = None
    output_dir: str = "ai_processing_results"
    log_level: str = "INFO"
    process_images: bool = False  # DISABLE IMAGE PROCESSING BY DEFAULT
    process_text: bool = True
    save_incremental: bool = True  # Save results after each batch
    
    def __post_init__(self):
        if self.priority_folders is None:
            self.priority_folders = [
                "~/Screenshots/",
                "~/Downloads/", 
                "~/Documents/",
                "~/Pictures/"
            ]
        
        # Handle custom priority folders from environment
        env_folders = os.getenv('PRIORITY_FOLDERS')
        if env_folders:
            self.priority_folders = [folder.strip() for folder in env_folders.split(',')]
        
        # Expand user paths
        self.priority_folders = [os.path.expanduser(folder) for folder in self.priority_folders]

@dataclass
class FileInfo:
    """Information about a discovered file"""
    path: str
    size: int
    file_type: str
    mime_type: str
    is_encrypted: bool = False
    priority: int = 0
    processed: bool = False
    tags: List[str] = None
    entities: List[str] = None
    locations: List[str] = None
    summary: str = ""  # NEW: AI-generated summary of the file
    error: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.entities is None:
            self.entities = []
        if self.locations is None:
            self.locations = []

class FileProcessor:
    """Main file processor class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        self.file_index: List[FileInfo] = []
        self.results_queue = queue.Queue()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize AI clients
        if not self.config.use_ollama:
            openai.api_key = self.config.openai_api_key or os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                raise ValueError("OpenAI API key not found in environment or config")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.output_dir}/processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_ollama_server(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                # Check if our model is available
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                # Check for exact match first
                if self.config.ollama_model in model_names:
                    return True
                
                # Check for partial match (e.g., llama3.2-vision might be stored as llama3.2-vision:latest)
                if any(self.config.ollama_model in name for name in model_names):
                    return True
                
                # Model not found
                self.logger.error(f"Model '{self.config.ollama_model}' not found!")
                self.logger.error(f"Available models: {model_names}")
                self.logger.error(f"Either:")
                self.logger.error(f"1. Change OLLAMA_MODEL in .env to one of the available models")
                self.logger.error(f"2. Or run: ollama pull {self.config.ollama_model}")
                return False
            
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Cannot connect to Ollama server: {e}")
            return False
    
    def start_ollama_server(self) -> bool:
        """Start Ollama server if not running"""
        if self.check_ollama_server():
            self.logger.info("Ollama server is already running with required model")
            return True
        
        self.logger.info("Starting Ollama server...")
        try:
            # Check if ollama command exists
            result = subprocess.run(["which", "ollama"], capture_output=True)
            if result.returncode != 0:
                self.logger.error("Ollama not found. Please install Ollama first:")
                self.logger.error("curl -fsSL https://ollama.ai/install.sh | sh")
                return False
            
            # Try to start ollama serve in background
            self.logger.info("Executing: ollama serve")
            process = subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Wait for server to start
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.check_ollama_server():
                    self.logger.info(f"Ollama server started successfully (took {i+1} seconds)")
                    return True
                
                # Check if process died
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    self.logger.error(f"Ollama server failed to start:")
                    self.logger.error(f"STDOUT: {stdout.decode()}")
                    self.logger.error(f"STDERR: {stderr.decode()}")
                    return False
            
            self.logger.error("Ollama server did not become ready within 30 seconds")
            self.logger.info("Try running manually: ollama serve")
            return False
            
        except Exception as e:
            self.logger.error(f"Error starting Ollama server: {e}")
            self.logger.info("Try starting manually: ollama serve")
            return False
    
    def discover_files(self) -> None:
        """Discover files in priority folders only"""
        self.logger.info("Starting file discovery in priority folders...")
        
        # Only scan priority folders
        for i, folder in enumerate(self.config.priority_folders):
            if os.path.exists(folder):
                self.logger.info(f"Scanning folder: {folder}")
                self._scan_directory(folder, priority=len(self.config.priority_folders) - i)
            else:
                self.logger.debug(f"Priority folder not found: {folder}")
        
        self.logger.info(f"Discovered {len(self.file_index)} files in priority folders")
    
    def _scan_directory(self, directory: str, priority: int = 0) -> None:
        """Scan a specific directory"""
        try:
            file_count = 0
            for root, dirs, files in os.walk(directory):
                # Skip hidden directories and common system/cache directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and 
                          d not in ['node_modules', '__pycache__', '.git', '.cache', 
                                   'Cache', 'cache', '.thumbnails', '.local/share/Trash']]
                
                for file in files:
                    # Skip hidden files and common cache/temp files
                    if file.startswith('.') or file.endswith(('.tmp', '.temp', '.log')):
                        continue
                        
                    filepath = os.path.join(root, file)
                    file_info = self._get_file_info(filepath, priority=priority)
                    if file_info:
                        self.file_index.append(file_info)
                        file_count += 1
                        
                        # Log progress every 100 files
                        if file_count % 100 == 0:
                            self.logger.debug(f"Found {file_count} files in {directory}")
                            
            self.logger.info(f"Found {file_count} files in {directory}")
                            
        except (PermissionError, OSError) as e:
            self.logger.debug(f"Cannot access {directory}: {e}")
    
    def _get_file_info(self, filepath: str, priority: int = 0) -> Optional[FileInfo]:
        """Get information about a specific file"""
        try:
            if not os.path.exists(filepath):
                return None
            
            stat = os.stat(filepath)
            if stat.st_size > self.config.max_file_size:
                return None
            
            # Detect file type
            mime_type = mimetypes.guess_type(filepath)[0] or "unknown"
            
            # Try to detect if file is encrypted
            is_encrypted = self._is_encrypted_file(filepath)
            
            # Determine file category
            file_type = self._categorize_file(filepath, mime_type)
            
            return FileInfo(
                path=filepath,
                size=stat.st_size,
                file_type=file_type,
                mime_type=mime_type,
                is_encrypted=is_encrypted,
                priority=priority
            )
            
        except (OSError, PermissionError) as e:
            self.logger.debug(f"Cannot access {filepath}: {e}")
            return None
    
    def _is_encrypted_file(self, filepath: str) -> bool:
        """Detect if a file is encrypted"""
        try:
            # Check file extension
            encrypted_extensions = {'.gpg', '.pgp', '.enc', '.aes', '.zip', '.7z', '.rar'}
            if Path(filepath).suffix.lower() in encrypted_extensions:
                return True
            
            # Check file magic
            try:
                file_magic = magic.from_file(filepath)
                encrypted_indicators = ['encrypted', 'pgp', 'gpg', 'password protected']
                return any(indicator in file_magic.lower() for indicator in encrypted_indicators)
            except:
                pass
            
            # Check file entropy (high entropy might indicate encryption)
            with open(filepath, 'rb') as f:
                data = f.read(1024)  # Read first 1KB
                if len(data) > 0:
                    entropy = self._calculate_entropy(data)
                    return entropy > 7.5  # High entropy threshold
            
        except Exception:
            pass
        
        return False
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        import math
        from collections import Counter
        
        if not data:
            return 0
        
        counter = Counter(data)
        entropy = 0
        data_len = len(data)
        
        for count in counter.values():
            p = count / data_len
            entropy -= p * math.log2(p)
        
        return entropy
    
    def _categorize_file(self, filepath: str, mime_type: str) -> str:
        """Categorize file type for processing"""
        path = Path(filepath)
        extension = path.suffix.lower()
        
        # Text files
        text_extensions = {'.txt', '.py', '.js', '.html', '.css', '.md', '.json', '.xml', '.yml', '.yaml', '.csv'}
        text_mimes = {'text/', 'application/json', 'application/xml'}
        
        if extension in text_extensions or any(mime_type.startswith(mime) for mime in text_mimes):
            return 'text'
        
        # Image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
        if extension in image_extensions or mime_type.startswith('image/'):
            return 'image'
        
        # Video files
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
        if extension in video_extensions or mime_type.startswith('video/'):
            return 'video'
        
        # Audio files
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        if extension in audio_extensions or mime_type.startswith('audio/'):
            return 'audio'
        
        return 'other'
    
    def organize_processing_queue(self) -> None:
        """Organize files into processing order"""
        self.logger.info("Organizing processing queue...")
        
        # Build list of enabled file types
        enabled_types = []
        if self.config.process_text:
            enabled_types.append('text')
        if self.config.process_images:
            enabled_types.append('image')
        
        if not enabled_types:
            self.logger.warning("No file types enabled for processing!")
            self.processing_queue = []
            return
        
        # Filter processable files based on enabled types
        processable_files = [
            f for f in self.file_index 
            if f.file_type in enabled_types and not f.is_encrypted
        ]
        
        # Log what was skipped
        if not self.config.process_images:
            image_files = [f for f in self.file_index if f.file_type == 'image']
            if image_files:
                self.logger.info(f"Skipping {len(image_files)} image files (image processing disabled)")
        
        if not self.config.process_text:
            text_files = [f for f in self.file_index if f.file_type == 'text']
            if text_files:
                self.logger.info(f"Skipping {len(text_files)} text files (text processing disabled)")
        
        # Sort by priority (higher first), then by file type for batching
        processable_files.sort(key=lambda x: (-x.priority, x.file_type))
        
        self.processing_queue = processable_files
        self.logger.info(f"Queue organized: {len(processable_files)} files ready for processing")
    
    def process_files(self) -> None:
        """Process files in batches"""
        if self.config.use_ollama:
            if not self.start_ollama_server():
                self.logger.error("="*60)
                self.logger.error("OLLAMA SERVER ISSUE DETECTED")
                self.logger.error("="*60)
                self.logger.error("The Ollama server is not responding properly.")
                self.logger.error("")
                self.logger.error("Quick fixes to try:")
                self.logger.error("1. Kill existing processes: pkill -f ollama")
                self.logger.error("2. Wait 3 seconds: sleep 3")
                self.logger.error("3. Start fresh: ollama serve")
                self.logger.error("4. Or run the debug script: chmod +x debug_ollama.sh && ./debug_ollama.sh")
                self.logger.error("")
                self.logger.error("Alternative: Switch to OpenAI by setting USE_OLLAMA=false in .env")
                self.logger.error("="*60)
                return
        
        self.organize_processing_queue()
        
        # Group files by type for batch processing
        file_groups = {}
        for file_info in self.processing_queue:
            if file_info.file_type not in file_groups:
                file_groups[file_info.file_type] = []
            file_groups[file_info.file_type].append(file_info)
        
        # Process each group
        for file_type, files in file_groups.items():
            self.logger.info(f"Processing {len(files)} {file_type} files")
            self._process_file_batch(files, file_type)
        
        # Save results
        self.save_results()
    
    def _process_file_batch(self, files: List[FileInfo], file_type: str) -> None:
        """Process a batch of files of the same type"""
        batch = []
        
        for file_info in files:
            batch.append(file_info)
            
            if len(batch) >= self.config.batch_size:
                self._process_batch(batch, file_type)
                if self.config.save_incremental:
                    self._save_incremental_results()  # Save after each batch
                batch = []
        
        # Process remaining files
        if batch:
            self._process_batch(batch, file_type)
            if self.config.save_incremental:
                self._save_incremental_results()  # Save after final batch
    
    def _process_batch(self, batch: List[FileInfo], file_type: str) -> None:
        """Process a single batch"""
        self.logger.info(f"Processing batch of {len(batch)} {file_type} files")
        
        for file_info in batch:
            try:
                if file_type == 'text':
                    self._process_text_file(file_info)
                elif file_type == 'image':
                    self._process_image_file(file_info)
                
                file_info.processed = True
                
                # Log the results for this file
                filename = os.path.basename(file_info.path)
                summary_preview = file_info.summary[:80] + "..." if len(file_info.summary) > 80 else file_info.summary
                self.logger.info(f"✅ {filename}: {summary_preview}")
                
                if file_info.tags:
                    self.logger.debug(f"   Tags: {', '.join(file_info.tags[:5])}")  # Show first 5 tags
                
            except Exception as e:
                file_info.error = str(e)
                filename = os.path.basename(file_info.path)
                self.logger.error(f"❌ {filename}: {e}")
    
    def _process_text_file(self, file_info: FileInfo) -> None:
        """Process a text file"""
        try:
            with open(file_info.path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if len(content.strip()) == 0:
                return
            
            # Truncate very long content
            if len(content) > 4000:
                content = content[:4000] + "... [truncated]"
            
            prompt = f"""Analyze this text file and provide:
1. A brief summary of what this file is about (1-2 sentences)
2. Main topics and themes (as tags)
3. Any mentioned locations
4. Important entities (people, organizations, etc.)

File: {os.path.basename(file_info.path)}
Content:
{content}

Respond in JSON format:
{{
    "summary": "Brief description of what this file contains or does",
    "tags": ["tag1", "tag2"],
    "locations": ["location1", "location2"],
    "entities": ["entity1", "entity2"]
}}"""
            
            result = self._call_ai_model(prompt, is_vision=False)
            self._parse_ai_response(file_info, result)
            
        except Exception as e:
            raise Exception(f"Text processing error: {e}")
    
    def _process_image_file(self, file_info: FileInfo) -> None:
        """Process an image file"""
        try:
            # Check if image is valid
            with Image.open(file_info.path) as img:
                img.verify()
            
            # Encode image for AI processing
            with open(file_info.path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            prompt = """Analyze this image and provide:
1. A brief summary describing what you see in the image (1-2 sentences)
2. What objects, people, or scenes do you see? (as tags)
3. Any locations you can identify or geographical features
4. Entities like people, brands, text, or organizations visible

Respond in JSON format:
{
    "summary": "Brief description of what this image shows",
    "tags": ["tag1", "tag2"],
    "locations": ["location1", "location2"],
    "entities": ["entity1", "entity2"]
}"""
            
            result = self._call_ai_model(prompt, is_vision=True, image_data=image_data)
            self._parse_ai_response(file_info, result)
            
        except Exception as e:
            raise Exception(f"Image processing error: {e}")
    
    def _call_ai_model(self, prompt: str, is_vision: bool = False, image_data: str = None) -> str:
        """Call the AI model (Ollama or OpenAI)"""
        if self.config.use_ollama:
            return self._call_ollama(prompt, is_vision, image_data)
        else:
            return self._call_openai(prompt, is_vision, image_data)
    
    def _call_ollama(self, prompt: str, is_vision: bool = False, image_data: str = None) -> str:
        """Call Ollama API"""
        url = f"{self.config.ollama_url}/api/generate"
        
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False
        }
        
        if is_vision and image_data:
            payload["images"] = [image_data]
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama server - is it running?")
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out (>120s)")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise Exception(f"Ollama API endpoint not found. Check if model '{self.config.ollama_model}' exists: ollama list")
            else:
                raise Exception(f"Ollama HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama request failed: {e}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from Ollama")
        except Exception as e:
            raise Exception(f"Unexpected Ollama error: {e}")
    
    def _call_openai(self, prompt: str, is_vision: bool = False, image_data: str = None) -> str:
        """Call OpenAI API"""
        messages = []
        
        if is_vision and image_data:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model=self.config.openai_model,
            messages=messages,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _parse_ai_response(self, file_info: FileInfo, response: str) -> None:
        """Parse AI response and update file info"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                file_info.summary = result.get('summary', 'No summary provided')
                file_info.tags = result.get('tags', [])
                file_info.locations = result.get('locations', [])
                file_info.entities = result.get('entities', [])
            else:
                # Fallback: treat as simple text response
                file_info.summary = response[:200] if response else "No summary available"  # Limit to 200 chars
                file_info.tags = ["processed"]
                
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            file_info.summary = response[:200] if response else "Processing completed"
            file_info.tags = ["processed"]
    
    def _save_incremental_results(self) -> None:
        """Save current processing results after each batch"""
        try:
            # Update CSV with current results
            self._save_csv_summary()
            
            # Also update JSON with current state
            results = {
                'processing_date': datetime.now().isoformat(),
                'config': asdict(self.config),
                'summary': {
                    'total_files_discovered': len(self.file_index),
                    'files_processed': len([f for f in self.file_index if f.processed]),
                    'files_with_summaries': len([f for f in self.file_index if f.summary and f.summary.strip()]),
                    'encrypted_files_found': len([f for f in self.file_index if f.is_encrypted]),
                    'errors': len([f for f in self.file_index if f.error]),
                    'status': 'processing'  # Indicate this is an incremental save
                },
                'files': [asdict(file_info) for file_info in self.file_index]
            }
            
            # Save incremental JSON results
            results_file = os.path.join(self.config.output_dir, 'processing_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Log progress
            processed_count = len([f for f in self.file_index if f.processed])
            total_processable = len([f for f in self.file_index if f.file_type in ['text', 'image'] and not f.is_encrypted])
            
            if total_processable > 0:
                progress_pct = (processed_count / total_processable) * 100
                self.logger.info(f"📊 Progress: {processed_count}/{total_processable} files ({progress_pct:.1f}%) - Results updated")
                
        except Exception as e:
            self.logger.warning(f"Could not save incremental results: {e}")
    
    def save_results(self) -> None:
        """Save final processing results"""
        results = {
            'processing_date': datetime.now().isoformat(),
            'config': asdict(self.config),
            'summary': {
                'total_files_discovered': len(self.file_index),
                'files_processed': len([f for f in self.file_index if f.processed]),
                'files_with_summaries': len([f for f in self.file_index if f.summary and f.summary.strip()]),
                'encrypted_files_found': len([f for f in self.file_index if f.is_encrypted]),
                'errors': len([f for f in self.file_index if f.error]),
                'status': 'completed'  # Mark as final/completed
            },
            'files': [asdict(file_info) for file_info in self.file_index]
        }
        
        # Save main results
        results_file = os.path.join(self.config.output_dir, 'processing_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary for easy viewing
        self._save_csv_summary()
        
        # Log summary statistics
        summary_stats = results['summary']
        self.logger.info("="*50)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info("="*50)
        self.logger.info(f"📁 Total files discovered: {summary_stats['total_files_discovered']}")
        self.logger.info(f"✅ Files processed: {summary_stats['files_processed']}")
        self.logger.info(f"📝 Files with summaries: {summary_stats['files_with_summaries']}")
        self.logger.info(f"🔐 Encrypted files found: {summary_stats['encrypted_files_found']}")
        self.logger.info(f"❌ Errors encountered: {summary_stats['errors']}")
        self.logger.info("="*50)
        self.logger.info(f"📊 Results saved to: {self.config.output_dir}")
        self.logger.info(f"📈 CSV summary: {self.config.output_dir}/summary.csv")
        self.logger.info(f"🔍 Full results: {self.config.output_dir}/processing_results.json")
    
    def _save_csv_summary(self) -> None:
        """Save a CSV summary of results"""
        import csv
        
        csv_file = os.path.join(self.config.output_dir, 'summary.csv')
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Path', 'Type', 'Size', 'Encrypted', 'Processed', 'Summary', 'Tags', 'Entities', 'Locations', 'Error'])
            
            for file_info in self.file_index:
                writer.writerow([
                    file_info.path,
                    file_info.file_type,
                    file_info.size,
                    file_info.is_encrypted,
                    file_info.processed,
                    file_info.summary,  # NEW: Include summary in CSV
                    '; '.join(file_info.tags),
                    '; '.join(file_info.entities),
                    '; '.join(file_info.locations),
                    file_info.error
                ])

def main():
    """Main execution function"""
    # Load configuration from environment
    config = Config(
        ollama_url=os.getenv('OLLAMA_URL', 'http://localhost:11434'),
        ollama_model=os.getenv('OLLAMA_MODEL', 'llama3.2-vision'),
        openai_api_key=os.getenv('OPENAI_API_KEY', ''),
        openai_model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
        use_ollama=os.getenv('USE_OLLAMA', 'true').lower() == 'true',
        max_file_size=int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024)),
        batch_size=int(os.getenv('BATCH_SIZE', 5)),
        output_dir=os.getenv('OUTPUT_DIR', 'ai_processing_results'),
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        process_images=os.getenv('PROCESS_IMAGES', 'false').lower() == 'true',
        process_text=os.getenv('PROCESS_TEXT', 'true').lower() == 'true',
        save_incremental=os.getenv('SAVE_INCREMENTAL', 'true').lower() == 'true'
    )
    
    processor = FileProcessor(config)
    
    try:
        processor.logger.info("Starting AI File Processor")
        processor.discover_files()
        processor.process_files()
        processor.logger.info("Processing completed successfully")
        
    except KeyboardInterrupt:
        processor.logger.info("Processing interrupted by user")
    except Exception as e:
        processor.logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()