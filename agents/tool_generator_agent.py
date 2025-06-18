#it's not properly detecting "alphavantage" as an available API because 
# the pattern matching is looking for exact matches, but the available 
# API is stored as 'alphavantage' while the detection patterns might be 
# finding 'alpha_vantage' (with underscore).
#tga8.py modified
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv, set_key
from api_key_config import API_KEY_MAPPING

import spacy
from spacy.matcher import Matcher
import logging

# Load environment variables
load_dotenv()

import re
from typing import Tuple, List, Set
from dataclasses import dataclass
from enum import Enum

class ValidationCategory(Enum):
    SECURITY = "security"
    FEASIBILITY = "feasibility"
    RESOURCE_INTENSIVE = "resource_intensive"
    INSUFFICIENT_DETAIL = "insufficient_detail"

@dataclass
class ValidationRule:
    """Represents a validation rule with patterns and category."""
    patterns: List[str]
    category: ValidationCategory
    message_template: str
    
class RequestValidator:
    """Domain-agnostic validator for user requests across any domain."""
    
    def __init__(self):
        self.validation_rules = self._initialize_rules()
        self._compiled_patterns = self._compile_patterns()
    
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize all validation rules in a structured format."""
        return [
            # Security violations - malicious intent
            ValidationRule(
                patterns=[
                    r'\b(hack|hacking|exploit|vulnerability|breach|crack)\b',
                    r'\bbreak\s+into\b',
                    r'\bunauthorized\s+access\b',
                    r'\bbypass\s+security\b',
                    r'\b(sql\s+injection|xss|csrf)\b',
                    r'\b(malware|virus|trojan|backdoor|keylogger)\b',
                    r'\b(phishing|ddos|botnet)\b',
                    r'\bsocial\s+engineering\b',
                    r'\b(password\s+cracking|brute\s+force)\b',
                    r'\bpenetration\s+test\b',
                    r'\b(reverse\s+engineer|decompile)\b',
                    r'\b(illegal|unauthorized|forbidden)\b'
                ],
                category=ValidationCategory.SECURITY,
                message_template="🚫 Security violation: Request involves '{match}' which relates to unauthorized access or potentially harmful activities."
            ),
            
            # Impossible/unrealistic expectations
            ValidationRule(
                patterns=[
                    r'\b(100%\s+accuracy|100%\s+accurate|never\s+fail|never\s+wrong)\b',
                    r'\b(guaranteed?\s+success|always\s+successful?)\b',
                    r'\b(perfect\s+prediction|crystal\s+ball)\b',
                    r'\b(time\s+travel|future\s+exactly)\b',
                    r'\b(infinite|unlimited)\b',
                    r'\bfree\s+.*(generator|money|premium)\b',
                    r'\bmagic\s+formula\b',
                    r'\bsecret\s+algorithm\s+that\s+always\s+wins\b',
                    r'\b(reads?\s+my\s+mind|mind\s+reading|telepathic|psychic)\b',
                    r'\bfoolproof\b'
                ],
                category=ValidationCategory.FEASIBILITY,
                message_template="🚫 Unrealistic expectation: '{match}' is not technically feasible. No system can guarantee perfect accuracy or impossible capabilities."
            ),
            
            # Resource-intensive operations requiring premium services
            ValidationRule(
                patterns=[
                    r'\b(analyze|process|scan)\s+(thousands?|hundreds?|millions?)\b',
                    r'\bscan\s+entire\b',
                    r'\b(bulk|mass)\s+(analysis|processing|screening)\b',
                    r'\bfull\s+(scan|dataset|analysis)\b',
                    r'\b(all|every)\s+(data|items?|records?)\b',
                    r'\bcomplete\s+analysis\b',
                    r'\b(real-?time|live)\s+(monitoring|tracking|analysis)\b',
                    r'\bhigh\s+frequency\s+(data|requests?)\b'
                ],
                category=ValidationCategory.RESOURCE_INTENSIVE,
                message_template="🚫 Resource-intensive operation: '{match}' requires premium services with higher rate limits and computational resources."
            )
        ]
    
    def _compile_patterns(self) -> dict:
        """Pre-compile regex patterns for better performance."""
        compiled = {}
        for rule in self.validation_rules:
            compiled[rule.category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in rule.patterns
            ]
        return compiled
    
    def _check_rule_category(self, request: str, category: ValidationCategory) -> Tuple[bool, str]:
        """Check if request violates rules in a specific category."""
        patterns = self._compiled_patterns[category]
        rule = next(rule for rule in self.validation_rules if rule.category == category)
        
        for pattern in patterns:
            match = pattern.search(request)
            if match:
                matched_text = match.group(0)
                return False, rule.message_template.format(match=matched_text)
        
        return True, ""
    
    def _check_request_detail(self, request: str) -> Tuple[bool, str]:
        """Check if request has sufficient detail."""
        cleaned_request = re.sub(r'\s+', ' ', request.strip())
        word_count = len(cleaned_request.split())
        
        if word_count < 5:
            return False, "🚫 Insufficient detail: Please provide a more detailed description of what you want to build."
        
        # Check for overly generic requests
        generic_patterns = [
            r'^(make|create|build|help)\s+(me\s+)?(a|an|some)?\s*$',
            r'^(i\s+need|i\s+want)\s+(help|assistance)\s*$',
            r'^(can\s+you|please)\s+(help|assist)\s*$'
        ]
        
        for pattern in generic_patterns:
            if re.match(pattern, cleaned_request, re.IGNORECASE):
                return False, "🚫 Too generic: Please specify what type of tool or functionality you need."
        
        return True, ""
    
    def validate_request(self, request: str) -> Tuple[bool, str, List[str]]:
        """
        Validate a user request comprehensively across all categories.
        
        Args:
            request: The user's request string
            
        Returns:
            Tuple[bool, str, List[str]]: (is_valid, reason_if_invalid, suggested_apis)
        """
        if not request or not request.strip():
            return False, "🚫 Empty request: Please provide a description of what you want to build.", []
        
        # Check each validation category
        for category in ValidationCategory:
            if category == ValidationCategory.INSUFFICIENT_DETAIL:
                continue  # Handle separately
                
            is_valid, error_message = self._check_rule_category(request, category)
            if not is_valid:
                return False, error_message, []
        
        # Check request detail last
        is_valid, error_message = self._check_request_detail(request)
        if not is_valid:
            return False, error_message, []
        
        return True, "", []
    
    def get_validation_summary(self, request: str) -> dict:
        """
        Get a detailed validation summary for debugging/analysis.
        
        Returns:
            dict: Summary of validation results by category
        """
        summary = {
            "request": request,
            "overall_valid": True,
            "categories": {}
        }
        
        for category in ValidationCategory:
            if category == ValidationCategory.INSUFFICIENT_DETAIL:
                is_valid, message = self._check_request_detail(request)
            else:
                is_valid, message = self._check_rule_category(request, category)
            
            summary["categories"][category.value] = {
                "valid": is_valid,
                "message": message if not is_valid else "✅ Passed"
            }
            
            if not is_valid:
                summary["overall_valid"] = False
        
        return summary
    
    def add_custom_rule(self, patterns: List[str], category: ValidationCategory, 
                       message_template: str) -> None:
        """
        Add a custom validation rule.
        
        Args:
            patterns: List of regex patterns to match
            category: ValidationCategory for the rule
            message_template: Template for error message (use {match} placeholder)
        """
        new_rule = ValidationRule(patterns, category, message_template)
        self.validation_rules.append(new_rule)
        
        # Recompile patterns
        if category not in self._compiled_patterns:
            self._compiled_patterns[category] = []
        
        self._compiled_patterns[category].extend([
            re.compile(pattern, re.IGNORECASE) for pattern in patterns
        ])

class APIRequirementAnalyzer:
    """Analyzes API requirements using LLM to determine if request is feasible."""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_request_feasibility(self, user_request: str, available_apis: Dict) -> Tuple[bool, str, List[str]]:
        """
        Use LLM to analyze if the request can be fulfilled with available APIs.
        
        Returns:
            Tuple[bool, str, List[str]]: (is_feasible, reason_if_not, required_apis)
        """
        
        # Create a prompt to analyze the request
        analysis_prompt = f"""
Analyze this user request for a tool: "{user_request}"

Available APIs and their capabilities:
{json.dumps(available_apis, indent=2)}

Determine if this request can be fulfilled. Consider:

1. Can this request be implemented using basic Python without external APIs?
2. If APIs are needed, which ones from the available list would be required?
3. If the request mentions APIs not in the available list, identify them as "unknown_apis"
4. Does it require specialized APIs that need special keys or paid subscriptions?
5. Does it require data-heavy operations that would exceed free API limits or quotas?
6. Does the request involve processing millions of records? (NOT FEASIBLE)
7. Would it exceed reasonable processing time limits (>5-10 minutes )?
8. Does it require access to large datasets that aren't publicly available?
9. Does it make impossible claims (100% accuracy, guaranteed success)?

Respond in this exact JSON format:
{{
    "feasible": true/false,
    "reason": "explanation if not feasible",
    "required_apis": ["list", "of", "known", "api", "names", "needed"],
    "unknown_apis": ["list", "of", "unknown", "api", "names", "mentioned"],
    "is_basic_python": true/false,
    "missing_capabilities": ["list", "of", "missing", "features"],
    "alternative_suggestion": "suggest alternatives if applicable"
}}

Note: 
- If the request can be implemented using basic Python without external APIs AND is reasonable in scope, set is_basic_python to true and leave required_apis empty.
- If the request mentions APIs not in the available list (like Twitter, Instagram, etc.), add them to unknown_apis
- Only mark as not feasible if the request is technically impossible, not just because APIs aren't configured
"""
        
        try:
            messages = [
                SystemMessage(content="You are an API and code analysis expert. Analyze the request and respond in the exact JSON format specified."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = self.model.invoke(messages)
            response_text = response.content
            
            # Extract JSON from response
            if "```json" in response_text:
                json_part = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_part = response_text.split("```")[1].strip()
            else:
                json_part = response_text
            
            # Parse the JSON response
            analysis = json.loads(json_part)
            
            # If it's a basic Python request, return success with empty API list
            if analysis.get("is_basic_python", False):
                return True, "", []
            
            # If there are unknown APIs, don't reject - let the main flow handle them
            if analysis.get("unknown_apis"):
                return True, "", analysis.get("required_apis", [])
        
            return (
                analysis.get("feasible", False),
                analysis.get("reason", "Analysis failed"),
                analysis.get("required_apis", [])
            )
            
        except Exception as e:
            # If LLM analysis fails, be conservative and reject complex requests
            return False, f"Unable to analyze request complexity: {str(e)}", []

class CodeGeneratorAgent:
    def __init__(self):
        """Initialize the Code Generator Agent with comprehensive validation."""
        # Configure Gemini API
        if not os.getenv('GOOGLE_API_KEY'):
            raise Exception("GOOGLE_API_KEY is required for the Code Generator Agent to function.")
        
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            convert_system_message_to_human=True
        )
        
        # Initialize validators
        self.validator = RequestValidator()
        self.api_analyzer = APIRequirementAnalyzer(self.model)
        
        # API Key Mapping
        self.api_key_mapping = API_KEY_MAPPING
        
        # Available APIs catalog with strict requirements
        self.available_apis = {
            'requests': {
                'description': 'HTTP requests for web APIs',
                'requires_key': False,
                'import_statement': 'import requests',
                'limitations': 'Rate limited by target API. No built-in rate limiting.',
                'capabilities': ['GET/POST requests', 'API calls', 'web scraping', 'data fetching'],
                'cannot_do': ['real-time streaming', 'websockets', 'complex authentication']
            },
            'openweather': {
                'description': 'Weather data and forecasts (free tier: 60 calls/minute)',
                'requires_key': True,
                'key_name': 'OPENWEATHER_API_KEY',
                'import_statement': 'import requests',
                'limitations': 'Free tier: 60 calls/minute, 1,000,000 calls/month',
                'capabilities': ['current weather', 'forecasts', 'historical data', 'weather alerts'],
                'cannot_do': ['real-time radar', 'satellite imagery', 'air quality data']
            },
            'newsapi': {
                'description': 'News headlines and articles (free tier: 100 requests/day)',
                'requires_key': True,
                'key_name': 'NEWSAPI_KEY',
                'import_statement': 'import requests',
                'limitations': 'Free tier: 100 requests/day, 1 month history max',
                'capabilities': ['news headlines', 'article snippets', 'source filtering'],
                'cannot_do': ['full article content', 'real-time news', 'social media data']
            },
            'google_genai': {
                'description': 'AI text generation and analysis using Google Gemini (requires API key)',
                'requires_key': True,
                'key_name': 'GOOGLE_API_KEY',
                'import_statement': 'from langchain_google_genai import ChatGoogleGenerativeAI',
                'model_config': {
                    'model': 'gemini-2.0-flash',
                    'temperature': 0.1,
                    'convert_system_message_to_human': True
                },
                'limitations': 'Rate limited, costs per token',
                'capabilities': ['text generation', 'analysis', 'summarization', 'classification'],
                'cannot_do': ['image generation', 'audio processing', 'real-time processing']
            },
            'email_smtp': {
                'description': 'Send emails via SMTP (Gmail)',
                'requires_key': True,
                'key_name': ['EMAIL_APP', 'EMAIL_APP_PASSWORD'],
                'import_statement': 'import smtplib\nfrom email.mime.text import MIMEText\nfrom email.mime.multipart import MIMEMultipart',
                'limitations': 'Gmail app passwords required, rate limits apply',
                'capabilities': ['send notifications', 'email reports', 'alerts'],
                'cannot_do': ['receive emails', 'email parsing', 'bulk emailing']
            },
            'alphavantage': {
                'description': 'Stock market data and financial indicators (free tier: 25 calls/day)',
                'requires_key': True,
                'key_name': 'ALPHAVANTAGE_KEY',
                'import_statement': 'import requests',
                'limitations': 'Free tier: 25 requests/day, 5 calls/minute',
                'capabilities': ['stock prices', 'technical indicators', 'forex data', 'crypto data', 'company fundamentals'],
                'cannot_do': ['real-time streaming', 'options data', 'news sentiment analysis']
            }
        }
        
        self.tools_directory = "dynamic_tools"
        self.registry_file = "tool_registry.json"
        self._ensure_directories()
        
        #Initialize NLP model for intelligent API detection**
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.matcher = Matcher(self.nlp.vocab)
            self._setup_api_patterns()
            print("✅ NLP model loaded successfully for intelligent API detection")
        except OSError:
            print("⚠️  spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            print("⚠️  Falling back to basic pattern matching...")
            self.nlp = None
            self.matcher = None
    
    def _ensure_directories(self):
        """Create necessary directories and files."""
        if not os.path.exists(self.tools_directory):
            os.makedirs(self.tools_directory)
        
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, 'w') as f:
                json.dump({"tools": []}, f)
                
        # Ensure the registry file has proper structure
        try:
            with open(self.registry_file, 'r') as f:
                registry = json.load(f)
                if "tools" not in registry:
                    registry["tools"] = []
                    with open(self.registry_file, 'w') as f:
                        json.dump(registry, f, indent=2)
        except (json.JSONDecodeError, KeyError):
            # If file is corrupted, recreate it
            with open(self.registry_file, 'w') as f:
                json.dump({"tools": []}, f, indent=2)
    
    def _get_api_name_from_mapping(self, detected_name: str) -> str:
        """Get the correct API name from API_KEY_MAPPING."""
        detected_upper = detected_name.upper().replace(' ', '_')
        
        # Check if this maps to a known API key
        if detected_upper in self.api_key_mapping:
            mapped_key = self.api_key_mapping[detected_upper]
            
            # Reverse lookup to find the API name in available_apis
            for api_name, config in self.available_apis.items():
                if config.get('requires_key'):
                    api_key = config.get('key_name')
                    if api_key == mapped_key or (isinstance(api_key, list) and mapped_key in api_key):
                        return api_name
        
        return detected_name.lower()
    
    def detect_api_requirements(self, user_request: str) -> List[str]:
        """Detect which APIs are needed and validate they're sufficient."""
        request_lower = user_request.lower()
        needed_apis = []
        
        # More precise API detection
        api_patterns = {
            'newsapi': ['news', 'headlines', 'articles', 'latest news', 'financial news',
                'trending news', 'top news', 'recent news', 'news headlines',
                'news articles', 'breaking news', 'market news', 'company news'],
            'openweather': ['weather', 'temperature', 'forecast', 'humidity', 'precipitation',
                'rain', 'snow', 'wind', 'climate', 'weather data', 'weather forecast',
                'weather readings', 'weather conditions', 'weather information',
                'daily weather', 'hourly weather', 'weather history', 'weather report'],
            'google_genai': ['ai', 'artificial intelligence', 'text generation', 'analysis', 
                'summarization', 'classification', 'language model', 'text processing',
                'natural language', 'nlp', 'text analysis', 'content generation'],
            'requests': ['http', 'web', 'scrape', 'fetch', 'api', 'calls', 'download',
                'get data', 'retrieve', 'web request', 'http request', 'rest api',
                'api call', 'web service', 'web api'],
            'email_smtp': [
                'send email', 'send mail', 'email me', 'mail me', 'send notification email',
                'sends email', 'sends mail', 'emails me', 'mails me', 'sends notification email',
                'email notification', 'send via email', 'email alert', 'mail alert',
                'emails notification', 'sends via email', 'emails alert', 'mails alert',
                'email the result', 'mail the result', 'send me an email', 'send me a mail',
                'sends me an email', 'sends me a mail', 'emails them', 'email it','email them', 
                'email report', 'mail report', 'send report via email', 'email summary',
                'emails report', 'mails report', 'sends report via email', 'emails summary',
                'mail summary', 'send summary via email', 'sends summary via email','via email'
            ],
            'alphavantage': [
                'stock', 'stocks', 'stock market', 'stock data', 'financial data',
                'alpha vantage', 'alphavantage', 'stock prices', 'market data', 'equity',
                'trading data', 'stock trading', 'financial market', 'ticker', 'symbol',
                'moving average', 'technical indicators', 'forex', 'crypto', 'cryptocurrency'
            ]
        }
        
        # Check for explicit API mentions
        api_indicators = ['api', 'service', 'data from', 'fetch from', 'get from', 'retrieve from']
        words = request_lower.split()
        
        for i, word in enumerate(words):
            if word in api_indicators and i + 1 < len(words):
                potential_api = words[i + 1].strip('.,!?')
                # USE API_KEY_MAPPING instead of normalize
                correct_api_name = self._get_api_name_from_mapping(potential_api)
                if correct_api_name in self.available_apis and correct_api_name not in needed_apis:
                    needed_apis.append(correct_api_name)
        
        # Check for direct API name mentions (flexible matching)
        for api_name in self.available_apis.keys():
            # Create variations of the API name to match against
            api_variations = [
                api_name,
                api_name.replace('_', ' '),
                api_name.replace('_', ''),
                api_name.lower(),
                api_name.upper()
            ]
            
            for variation in api_variations:
                if variation in request_lower and api_name not in needed_apis:
                    needed_apis.append(api_name)
                    break
        
        # Check ALL patterns for ALL APIs (don't return early)
        for api, patterns in api_patterns.items():
            for pattern in patterns:
                if pattern in request_lower:
                    if api not in needed_apis:  # Avoid duplicates
                        needed_apis.append(api)
                    break  # Found a match for this API, check next API
        
        # Check for pattern-based API detection
        for api_name, keywords in api_patterns.items():
            if any(keyword in request_lower for keyword in keywords):
                if api_name in self.available_apis:
                    needed_apis.append(api_name)
                            
        # Default to requests for web-related requests
        if not needed_apis and any(word in request_lower for word in ['web', 'scrape', 'fetch', 'api', 'http']):
            needed_apis.append('requests')
        
        # Remove duplicates and return    
        return list(set(needed_apis))
    
    def _setup_api_patterns(self):
        """Setup spaCy patterns for better API detection."""
        if not self.matcher:
            return
            
        # **CHANGE 3: Define intelligent patterns for API/service detection**
        patterns = [
            # Direct API mentions
            [{"LOWER": {"IN": ["api", "service", "platform"]}}, {"IS_ALPHA": True}],
            [{"IS_ALPHA": True}, {"LOWER": {"IN": ["api", "service", "platform"]}}],
            
            # Data source patterns
            [{"LOWER": {"IN": ["from", "scrape", "scraping", "fetch", "get"]}}, {"ENT_TYPE": "ORG"}],
            [{"LOWER": {"IN": ["from", "scrape", "scraping", "fetch", "get"]}}, {"IS_ALPHA": True, "IS_TITLE": True}],
            
            # Website/domain patterns  
            [{"LIKE_URL": True}],
            [{"IS_ALPHA": True}, {"TEXT": "."}, {"LOWER": {"IN": ["com", "org", "net", "io"]}}],
        ]
        
        try:
            self.matcher.add("API_SERVICE", patterns)
            print(f"✅ Added {len(patterns)} patterns to spaCy matcher")
        except Exception as e:
            print(f"⚠️  Error adding patterns to matcher: {e}")
            # Create a simpler fallback pattern
            fallback_patterns = [
                [{"IS_ALPHA": True}, {"LOWER": "api"}],
                [{"LOWER": "from"}, {"IS_TITLE": True}],
            ]
            try:
                self.matcher.add("API_SERVICE", fallback_patterns)
                print("✅ Added fallback patterns to matcher")
            except Exception as fallback_error:
                print(f"❌ Could not add any patterns: {fallback_error}")
                self.matcher = None  # Disable matcher if it can't work

    def _detect_unknown_apis_nlp(self, user_request: str) -> List[str]:
        """
        Use NLP to intelligently detect API/service names.
        Much more accurate than manual pattern matching.
        """
        if not self.nlp:
            # Fallback to basic detection if NLP not available
            return self._detect_unknown_apis(user_request)
        
        doc = self.nlp(user_request)
        unknown_apis = []
        
        # **CHANGE 4: Extract organizations using spaCy's pre-trained NER**
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:  # Organizations and Products
                api_name = ent.text.lower().strip()
                
                # Remove "api" suffix if present to avoid duplicates
                if api_name.endswith(' api'):
                    api_name = api_name[:-4].strip()
                elif api_name.endswith('api'):
                    api_name = api_name[:-3].strip()
                
                # USE API_KEY_MAPPING to check if this is a known API
                correct_api_name = self._get_api_name_from_mapping(api_name)
                if correct_api_name in self.available_apis:
                    # This is actually an available API, not unknown
                    continue

                # Replace spaces with underscores for multi-word API names
                api_name = api_name.replace(' ', '_')
                
                if (api_name not in self.available_apis and 
                    len(api_name) > 2 and
                    not self._is_generic_term(api_name) and
                    api_name not in unknown_apis):
                    unknown_apis.append(api_name)
        
        # Used spaCy Matcher for context-aware detection**
        if self.matcher:
            try:
                matches = self.matcher(doc)
                matched_tokens = set()  # Track tokens already processed
                
                for match_id, start, end in matches:
                    span = doc[start:end]
                    
                    # Extract the actual service name from the matched span
                    for token in span:
                        if token.i in matched_tokens:
                            continue
                        
                        if (token.is_alpha and 
                            len(token.text) > 2 and 
                            token.text.lower() not in self.available_apis and
                            not self._is_generic_term(token.text.lower())):
                            candidate = token.text.lower()
                            # Skip if it's just "api"
                            if candidate == 'api':
                                continue
                            if candidate not in unknown_apis:
                                unknown_apis.append(candidate)
                                matched_tokens.add(token.i)
            except Exception as e:
                print(f"⚠️  Matcher error: {e}. Skipping pattern matching.")
        
        # Additional context-based detection
        # Look for capitalized words in specific contexts, including compound names
        i = 0
        while i < len(doc):
            token = doc[i]
            if (token.is_title and 
                len(token.text) > 2 and
                i > 0 and 
                doc[i-1].lower_ in ["from", "via", "using", "scrape", "scraping"]):
                
                # Check if this starts a multi-word API name
                compound_name = [token.text.lower()]
                j = i + 1
                
                # Look ahead for additional capitalized words or "API"
                while j < len(doc) and j < i + 3:  # Limit to 3 words max
                    next_token = doc[j]
                    if (next_token.is_title or next_token.lower_ == 'api') and len(next_token.text) > 1:
                        if next_token.lower_ != 'api':
                            compound_name.append(next_token.text.lower())
                        j += 1
                    else:
                        break
                
                # Create the compound API name
                if len(compound_name) > 1 or len(compound_name[0]) > 3:
                    candidate = '_'.join(compound_name)
                    if (candidate not in self.available_apis and 
                        candidate not in unknown_apis and
                        not self._is_generic_term(candidate)):
                        unknown_apis.append(candidate)
                    i = j  # Skip processed tokens
                else:
                    candidate = compound_name[0]
                    if (candidate not in self.available_apis and 
                        candidate not in unknown_apis and
                        not self._is_generic_term(candidate)):
                        unknown_apis.append(candidate)
                    i += 1
            else:
                i += 1
        
        return list(set(unknown_apis))
    
    def _is_generic_term(self, word: str) -> bool:
        """
        Check if word is a generic term that shouldn't be treated as API.
        Much smaller and focused list compared to manual approach.
        """
        generic_terms = {
            # Data types
            'data', 'information', 'content', 'text', 'json', 'xml', 'csv',
            # Common actions  
            'scraping', 'scrape', 'fetch', 'get', 'retrieve', 'extract',
            # Generic objects
            'listings', 'prices', 'descriptions', 'results', 'details',
            # Tech terms
            'api', 'service', 'platform', 'website', 'site', 'web',
            # Common adjectives
            'real', 'estate', 'financial', 'social', 'media'
        }
        return word.lower() in generic_terms
    
    def _detect_unknown_apis(self, user_request: str) -> List[str]:
        """Detect potential API names that are not in our available_apis catalog."""
        request_lower = user_request.lower()
        unknown_apis = []
        
        # API detection patterns
        api_patterns = [
            r'from\s+([\w\s]+?)\s+api',
            r'using\s+([\w\s]+?)\s+api',
            r'via\s+([\w\s]+?)\s+api', 
            r'with\s+([\w\s]+?)\s+api',
            r'([\w\s]+?)\s+api(?:\s|$|[.,!?])',
            r'get.*from\s+([\w\s]+?)(?:\s+api|\s|$|[.,!?])',
            r'fetch.*from\s+([\w\s]+?)(?:\s+api|\s|$|[.,!?])',
            r'retrieve.*from\s+([\w\s]+?)(?:\s+api|\s|$|[.,!?])',
            r'access\s+([\w\s]+?)(?:\s+api|\s|$|[.,!?])',
            r'connect\s+to\s+([\w\s]+?)(?:\s+api|\s|$|[.,!?])'
        ]
        
        # Find potential API names using regex patterns
        for pattern in api_patterns:
            matches = re.findall(pattern, request_lower)
            for match in matches:
                # Clean the match - remove common words and normalize
                cleaned_match = match.strip().lower()
                # Remove trailing words that are too generic
                words = cleaned_match.split()
                filtered_words = []
                for word in words:
                    if word not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'api', 'data', 'service']:
                        filtered_words.append(word)
                    else:
                        break  # Stop at first generic word
                
                if filtered_words:
                    api_name = '_'.join(filtered_words)
                    #Normalize before checking
                    normalized_api = self._get_api_name_from_mapping(api_name)
                    if normalized_api in self.available_apis:
                        # This is actually an available API, not unknown
                        continue
                    
                    if (api_name not in self.available_apis and 
                        len(api_name) > 2 and 
                        api_name not in unknown_apis):
                        unknown_apis.append(api_name)
        
        # Enhanced explicit mentions detection for compound names
        words = request_lower.split()
        i = 0
        
        while i < len(words):
            word = words[i]
            if word in ['from', 'using', 'via', 'with'] and i + 1 < len(words):
                # Look for multi-word API names
                api_words = []
                j = i + 1
                
                # Collect consecutive words until we hit "api" or reach end/punctuation
                while j < len(words) and j < i + 4:  # Limit to 3 words after trigger
                    next_word = words[j].strip('.,!?')
                    if next_word == 'api':
                        break
                    elif (len(next_word) > 2 and 
                        next_word not in ['the', 'a', 'an', 'and', 'or', 'but', 'data', 'service']):
                        api_words.append(next_word)
                        j += 1
                    else:
                        break
                
                if api_words:
                    potential_api = '_'.join(api_words)
                    if (potential_api not in self.available_apis and 
                        potential_api not in unknown_apis and
                        len(potential_api) > 2):
                        unknown_apis.append(potential_api)
                    i = j
                else:
                    i += 1
            else:
                i += 1

            # Remove duplicates and clean up
            unique_apis = []
            for api in unknown_apis:
                # Normalize API names (remove redundant "api" suffixes)
                normalized = api.lower().strip()
                if normalized.endswith(' api'):
                    normalized = normalized[:-4].strip()
                elif normalized.endswith('api') and len(normalized) > 3:
                    normalized = normalized[:-3].strip()
                    
                if normalized not in unique_apis and normalized != 'api':
                    unique_apis.append(normalized)
                    
        return unique_apis 
    
    def _analyze_api_requirements(self, user_request: str, unknown_apis: List[str]) -> Dict[str, str]:
        """
        Analyze which APIs are required vs optional for the given request.
        Returns dict with api_name -> 'required'/'optional'
        """
        if not unknown_apis:
            return {}
        
        try:
            #More intelligent prompt using NLP insights**
            if self.nlp:
                doc = self.nlp(user_request)
                main_verb = None
                main_object = None
                
                # Extract main action and object for better analysis
                for token in doc:
                    if token.pos_ == "VERB" and not main_verb:
                        main_verb = token.lemma_
                    if token.ent_type_ == "ORG" and not main_object:
                        main_object = token.text
                
                context = f"Main action: {main_verb or 'unknown'}, Main target: {main_object or 'unknown'}"
            else:
                context = "Context analysis not available"
            
            analysis_prompt = f"""
    Analyze this user request: "{user_request}"
    {context}
    Unknown APIs detected: {unknown_apis}

    For each API, determine if it's REQUIRED (without it, the core functionality cannot work) or OPTIONAL (nice to have, but request can work without it).

    Consider:
    - If the request is specifically mentions a service by name for the main task (e.g., "Twitter posts"), that API is REQUIRED
    - If the request mentions multiple services but focuses on one, others might be OPTIONAL
    - If mentioned for additional features or alternatives then OPTIONAL
    - If an API provides supplementary features, it's OPTIONAL
    - If unsure, lean towards REQUIRED for better user experience

    Respond in this exact JSON format:
    {{
        "api_requirements": {{
            "api_name": "required/optional",
            "api_name2": "required/optional"
        }}
    }}
    """
            
            messages = [
                SystemMessage(content="You are an API requirement analyzer. Analyze which APIs are required vs optional."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = self.model.invoke(messages)
            response_text = response.content
            
            # Extract JSON from response
            if "```json" in response_text:
                json_part = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_part = response_text.split("```")[1].strip()
            else:
                json_part = response_text
            
            analysis = json.loads(json_part)
            return analysis.get("api_requirements", {})
            
        except Exception as e:
            print(f"⚠️  Could not analyze API requirements: {e}")
            # Default: assume all are required
            return {api: "required" for api in unknown_apis}
    
    def _configure_new_api(self, api_name: str, is_required: bool = True) -> bool:
        """Configure a new API by prompting user for details."""
        requirement_text = "🔴 REQUIRED" if is_required else "🟡 OPTIONAL"
        print(f"\n🔧 Configuring {requirement_text} API: {api_name}")
        
        if not is_required:
            skip_choice = input(f"This API is optional. Skip {api_name} configuration? (y/n): ").strip().lower()
            if skip_choice == 'y':
                print(f"⏭️  Skipped {api_name} API configuration")
                return False
        
        print("Please provide the following information:")
        
        # Get API key
        key_name = f"{api_name.upper()}_API_KEY"
        print(f"\n💡 For {api_name} API, you'll typically need:")
        print(f"  - API Key or Access Token")
        print(f"  - Check {api_name} developer documentation for exact requirements")
    
        key_value = input(f"Enter {key_name}: ").strip()
        
        if not key_value:
            if is_required:
                print(f"❌ {api_name} API key is REQUIRED but not provided!")
                return False
            else:
                print(f"⏭️  Skipped {api_name} API (no key provided)")
                return False
        
        # Get optional description
        description = input(f"Enter description for {api_name} (optional): ").strip()
        if not description:
            description = f'{api_name} API integration'
        
        # Add the new API to available_apis
        self.available_apis[api_name] = {
            'description': description,
            'requires_key': True,
            'key_name': key_name,
            'import_statement': 'import requests',
            'limitations': 'Rate limited by API provider',
            'capabilities': ['data fetching', 'API integration'],
            'cannot_do': ['premium features', 'enterprise features']
        }
        
        # Set the API key in current environment
        os.environ[key_name] = key_value
        print(f"✅ {api_name} API has been configured successfully!")
        
        # ACTUALLY save to .env file
        set_key('.env', key_name, key_value)
        print(f"✅ {key_name} has been configured and saved to .env!")

        return True
    
    def _prompt_for_missing_api_keys(self, missing_keys: List[str]) -> bool:
        """
        Prompt user for missing API keys with better UX.
        Returns True if all keys were provided, False otherwise.
        """
        if not missing_keys:
            return True
        
        print(f"\n🔑 The following API keys are required for your request:")
        for key in missing_keys:
            # Find which API this key belongs to
            api_name = None
            for api, config in self.available_apis.items():
                if config.get('requires_key') and config.get('key_name') == key:
                    api_name = api
                    break
            
            if api_name:
                print(f"  • {key} (for {api_name} - {self.available_apis[api_name]['description']})")
            else:
                print(f"  • {key}")
        
        print(f"\n💡 You can:")
        print(f"  1. Provide the API keys now")
        print(f"  2. Cancel and modify your request to use different APIs")
        print(f"  3. Skip APIs that require keys (limited functionality)")
        
        choice = input("\nWhat would you like to do? (1/2/3): ").strip()
        
        if choice == '1':
            print("\nPlease provide the API keys:")
            successfully_added = []
            
            for key in missing_keys:
                # Find API name for better context
                api_name = None
                for api, config in self.available_apis.items():
                    if config.get('requires_key') and config.get('key_name') == key:
                        api_name = api
                        break
                
                if api_name:
                    print(f"\n🔧 {api_name} API Configuration:")
                    print(f"   Description: {self.available_apis[api_name]['description']}")
                    print(f"   Limitations: {self.available_apis[api_name]['limitations']}")
                
                value = input(f"Enter {key} (or press Enter to skip): ").strip()
                
                if value:
                    os.environ[key] = value
                    successfully_added.append(key)
                    print(f"✅ {key} configured successfully!")
                else:
                    print(f"⚠️  Skipped {key}")
            
            if successfully_added:
                print(f"\n✅ Successfully configured {len(successfully_added)} API key(s)")
                return True
            else:
                print(f"\n⚠️  No API keys were provided")
                return False
                
        elif choice == '2':
            return False
        elif choice == '3':
            print("⚠️  Proceeding without API keys - functionality will be limited")
            return True
        else:
            print("❌ Invalid choice")
            return False
    
    def _check_and_configure_apis(self, user_request: str) -> Tuple[bool, List[str], str]:
        """
        Check for required APIs and configure missing ones if needed.
        Returns: (success, detected_apis, error_message)
        """
        # Step 1: Detect known APIs
        detected_apis = self.detect_api_requirements(user_request)
        print(f"🔍 Detected APIs needed: {detected_apis if detected_apis else 'None (Basic Python)'}")
        
        # Step 2: Detect unknown APIs
        unknown_apis = self._detect_unknown_apis_nlp(user_request)
        
        if unknown_apis:
            print(f"\n⚠️  Detected potential additional APIs: {', '.join(unknown_apis)}")
        
            # **Filter out unknown APIs that are already covered by detected APIs**
            uncovered_unknown_apis = []
            for unknown_api in unknown_apis:
                # USE API_KEY_MAPPING to get correct API name
                correct_api_name = self._get_api_name_from_mapping(unknown_api)
                
                # Check if this unknown API is already in detected APIs (direct match)
                if correct_api_name in detected_apis:
                    print(f"✅ {unknown_api} is already detected as {correct_api_name}")
                    continue
                
                # Check if this unknown API is already covered by detected APIs
                is_covered = False
                for detected_api in detected_apis:
                    api_config = self.available_apis.get(detected_api, {})
                    capabilities = api_config.get('capabilities', [])
                    
                    # Check if the unknown API functionality is covered by existing detected APIs
                    if (correct_api_name == detected_api or
                        unknown_api.lower() in [cap.lower() for cap in capabilities] or 
                        unknown_api.lower() in detected_api.lower() or
                        self._is_functionality_covered(unknown_api, detected_api, api_config)):
                        is_covered = True
                        print(f"✅ {unknown_api} functionality is already covered by {detected_api}")
                        break
                
                if not is_covered:
                    uncovered_unknown_apis.append(unknown_api)
            
            # **Only proceed if there are truly uncovered APIs**
            if uncovered_unknown_apis:
                print(f"\n🔧 APIs that need configuration: {', '.join(uncovered_unknown_apis)}")
                
                # Analyze which APIs are required vs optional (only for uncovered APIs)
                api_requirements = self._analyze_api_requirements(user_request, uncovered_unknown_apis)
                
                # Show requirements breakdown
                required_apis = [api for api, req in api_requirements.items() if req == 'required']
                optional_apis = [api for api, req in api_requirements.items() if req == 'optional']
                
                if required_apis:
                    print(f"\n🔴 REQUIRED APIs: {', '.join(required_apis)}")
                    print("   (These are essential - code cannot work without them)")
                
                if optional_apis:
                    print(f"\n🟡 OPTIONAL APIs: {', '.join(optional_apis)}")
                    print("   (These provide additional features but are not essential)")
                    
                # Ask user if they want to configure these APIs
                configure_choice = input("\nWould you like to configure these additional APIs? (y/n): ").strip().lower()
                
                if configure_choice == 'y':
                    successfully_configured = []
                    failed_required = []
                    
                    # Configure APIs in order of importance (required first)
                    all_apis_to_configure = required_apis + optional_apis
            
                    for api in all_apis_to_configure:
                        is_required = api in required_apis
                        
                        if self._configure_new_api(api, is_required):
                            successfully_configured.append(api)
                            detected_apis.append(api)  # Add to detected APIs
                        elif is_required:
                            failed_required.append(api)
                    
                    # Check if any required APIs failed
                    if failed_required:
                        return False, detected_apis, f"Required APIs not configured: {', '.join(failed_required)}. Cannot generate code without these APIs."
                    
                    if successfully_configured:
                        print(f"\n✅ Successfully configured: {', '.join(successfully_configured)}")
                    else:
                        print(f"\n⚠️  No additional APIs were configured")
                else:
                    # Check if there are required APIs that user chose not to configure
                    if required_apis:
                        return False, detected_apis, f"Required APIs ({', '.join(required_apis)}) not configured. Cannot generate code without these APIs."
                    else:
                        print("⚠️  Continuing without additional APIs...")
            else:
                print("✅ All detected API functionality is already covered by known APIs")
        
        # Step 3: Check if all required API keys are available
        missing_keys, existing_keys = self._check_api_keys(detected_apis)
        
        if missing_keys:
            print(f"\n🔑 Missing required API keys: {', '.join(missing_keys)}")
            
            # Use enhanced prompting method
        if not self._prompt_for_missing_api_keys(missing_keys):
            return False, detected_apis, "API key configuration was cancelled or incomplete."
        
        return True, detected_apis, ""
    
    # **CHANGE 3: Add new helper method to check functionality coverage**
    def _is_functionality_covered(self, unknown_api: str, detected_api: str, api_config: Dict) -> bool:
        """
        Check if the unknown API functionality is covered by a detected API.
        This makes the system more domain-agnostic by checking semantic similarity.
        """
        unknown_lower = unknown_api.lower()
        detected_lower = detected_api.lower()
        
        # Direct name matching
        if unknown_lower in detected_lower or detected_lower in unknown_lower:
            return True
        
        # Check capabilities
        capabilities = [cap.lower() for cap in api_config.get('capabilities', [])]
        if any(unknown_lower in cap or cap in unknown_lower for cap in capabilities):
            return True
        
        # **CHANGE 4: Domain-agnostic semantic coverage mapping**
        coverage_mappings = {
            # Social media APIs
            'twitter': ['social', 'posts', 'tweets', 'social media'],
            'facebook': ['social', 'posts', 'social media'],
            'instagram': ['social', 'posts', 'photos', 'social media'],
            'linkedin': ['professional', 'social', 'posts', 'social media'],
            
            # Communication APIs  
            'slack': ['messaging', 'chat', 'notifications', 'communication'],
            'discord': ['messaging', 'chat', 'notifications', 'communication'],
            'telegram': ['messaging', 'chat', 'notifications', 'communication'],
            'whatsapp': ['messaging', 'chat', 'notifications', 'communication'],
            
            # Finance APIs
            'stripe': ['payment', 'billing', 'finance', 'transactions'],
            'paypal': ['payment', 'billing', 'finance', 'transactions'],
            'alpha_vantage': ['stocks', 'finance', 'market data', 'financial'],
            'yahoo_finance': ['stocks', 'finance', 'market data', 'financial'],
            
            # Cloud/Storage APIs
            'aws': ['cloud', 'storage', 'compute', 'infrastructure'],
            'gcp': ['cloud', 'storage', 'compute', 'infrastructure'],
            'azure': ['cloud', 'storage', 'compute', 'infrastructure'],
            'dropbox': ['storage', 'files', 'cloud storage'],
            'google_drive': ['storage', 'files', 'cloud storage'],
            
            # AI/ML APIs
            'openai': ['ai', 'text generation', 'language model', 'artificial intelligence'],
            'huggingface': ['ai', 'machine learning', 'models', 'artificial intelligence'],
            'anthropic': ['ai', 'text generation', 'language model', 'artificial intelligence'],
            
            # Maps/Location APIs
            'google_maps': ['maps', 'location', 'geocoding', 'navigation'],
            'mapbox': ['maps', 'location', 'geocoding', 'navigation'],
            
            # Generic patterns
            'database': ['data', 'storage', 'persistence'],
            'cache': ['caching', 'performance', 'storage'],
            'queue': ['messaging', 'async', 'background tasks'],
        }
        
        # Check if detected API can cover unknown API functionality
        for known_api, functions in coverage_mappings.items():
            if known_api in detected_lower:
                if any(func in unknown_lower for func in functions):
                    return True
        
        # Check reverse mapping - if unknown API is in our mappings and detected API covers those functions
        if unknown_lower in coverage_mappings:
            unknown_functions = coverage_mappings[unknown_lower]
            if any(func in detected_lower or func in cap for func in unknown_functions for cap in capabilities):
                return True
        
        return False    
        
    def _basic_validation(self, user_request: str) -> Tuple[bool, str]:
        """Perform comprehensive validation of the user request."""
        
        # Step 1: Basic keyword validation
        is_valid, reason, _ = self.validator.validate_request(user_request)
        if not is_valid:
            print("\n ❌ Step 1 validation failed ")
            summary = self.validator.get_validation_summary(user_request)
            print(f"Request: {summary['request']}")
            print(f"Overall Valid: {summary['overall_valid']}")
            print("\nCategory Results:")
            for category, result in summary['categories'].items():
                print(f"  {category}: {result['message']}")
            return False, reason
        
        # Step 2: LLM-based feasibility analysis
        try:
            is_feasible, feasibility_reason, _ = self.api_analyzer.analyze_request_feasibility(
                user_request, self.available_apis
            )
            if not is_feasible:
                return False, f"🤖 AI Analysis: {feasibility_reason}"
        except Exception as e:
            print(f"⚠️  Warning: Could not perform AI analysis: {e}")
            # Continue without AI analysis
        
        return True, ""
    
    def _check_api_keys(self, apis: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Check which required API keys are missing and collect existing ones."""
        missing_keys = []
        existing_keys = {}
        
        for api in apis:
            if api in self.available_apis and self.available_apis[api]['requires_key']:
                key_name = self.available_apis[api]['key_name']
                
                if isinstance(key_name, list):
                    for key in key_name:
                        if os.getenv(key):
                            existing_keys[key] = os.getenv(key)
                        else:
                            missing_keys.append(key)
                else:
                    if os.getenv(key_name):
                        existing_keys[key_name] = os.getenv(key_name)
                    else:
                        missing_keys.append(key_name)
        
        return missing_keys, existing_keys

    def create_tool(self, user_request: str) -> Dict:
        """ 
        Main method that follows the specified flow:
        1. User enters query
        2. Analyze query and check API requirements
        2.a If no APIs required -> create code
        2.b If APIs required -> check availability and configure if needed
        """        
        try:
            print(f"🔧 Processing request: {user_request}")
            
            # Step 1: Basic validation
            print("🔍 Running basic validation...")
            is_valid, error_reason = self._basic_validation(user_request)
            if not is_valid:
                return {
                    "status": "rejected",
                    "message": f"❌ Request rejected: {error_reason}"
                }
            
            print("✅ Basic validation passed")
            
            # Step 2: Check and configure APIs
            print("🔍 Analyzing API requirements...")
            api_success, detected_apis, api_error = self._check_and_configure_apis(user_request)
            
            if not api_success:
                return {
                    "status": "cancelled",
                    "message": f"❌ API configuration failed: {api_error}",
                    "detected_apis": detected_apis
                }
            
            print(f"✅ API check completed. Final APIs to use: {', '.join(detected_apis) if detected_apis else 'None (Basic Python)'}")
            
            # Step 3: Show API limitations if any APIs are being used
            if detected_apis:
                print("\n⚠️  API LIMITATIONS:")
                for api in detected_apis:
                    if api in self.available_apis:
                        print(f"- {api}: {self.available_apis[api]['limitations']}")
                
                # Confirm user wants to proceed
                proceed = input("\nDo you want to proceed with these API limitations? (y/n): ").strip().lower()
                if proceed != 'y':
                    return {
                        "status": "cancelled",
                        "message": "Tool creation cancelled due to API limitations."
                    }
            
            print("\nDetected APis : ",detected_apis)
            # Step 4: Generate code
            print("\n🚀 Generating code...")
            generated_code, final_apis = self.generate_tool_code(user_request, detected_apis)        
            print('\n 🌟🌟 Generated code : \n', generated_code)
            
            # Extract function name
            function_name = self.extract_function_name(generated_code)
            
            # Show code to user for final approval
            print("\n" + "="*50)
            print("GENERATED CODE:")
            print("="*50)
            print(generated_code)
            print("="*50)
            print(f"Function Name: {function_name}")
            print(f"APIs Used: {', '.join(final_apis) if final_apis else 'None (Basic Python)'}")
            print("="*50)
            
            # Final approval
            approval = input("\nApprove this code? (y/n): ").strip().lower()
            
            if approval == 'y':
                filepath = self.save_tool(generated_code, function_name, user_request, final_apis)
                
                result = {
                    "status": "success",
                    "function_name": function_name,
                    "filepath": filepath,
                    "apis_used": final_apis,
                    "message": f"Tool '{function_name}' created successfully!"
                }
                
                print(f"\n✅ {result['message']}")
                print(f"📁 Saved to: {filepath}")
                self._show_usage_reminders()
                
                return result
            else:
                return {
                    "status": "cancelled",
                    "message": "Tool creation cancelled by user."
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating tool: {str(e)}"
            }

    def generate_tool_code(self, user_request: str, detected_apis: List[str]) -> Tuple[str, List[str]]:
        """Generate code only for validated requests with confirmed APIs."""
        
        # Create enhanced prompt with strict API usage
        api_info = ""
        if detected_apis:
            for api in detected_apis:
                if api in self.available_apis:
                    api_data = self.available_apis[api]
                    api_info += f"\n- {api}: {api_data['description']}"
                    api_info += f"\n  Capabilities: {', '.join(api_data['capabilities'])}"
                    api_info += f"\n  Limitations: {api_data['limitations']}"
                    if api == 'google_genai':
                        api_info += f"\n  Import: from langchain_google_genai import ChatGoogleGenerativeAI"
                        api_info += f"\n  Model: gemini-2.0-flash"
                    else:
                        api_info += f"\n  Import: {api_data['import_statement']}"
                    if api_data['requires_key']:
                        if isinstance(api_data['key_name'], list):
                            keys = ", ".join([f"os.getenv('{key}')" for key in api_data['key_name']])
                            api_info += f"\n  Keys: {keys}"
                        else:
                            api_info += f"\n  Key: os.getenv('{api_data['key_name']}')"
        else:
            print("\nNO APIs detected\n")
        
        prompt = f"""
You are creating a Python function for: "{user_request}"

STRICT REQUIREMENTS:
1. If APIs are needed, use ONLY these verified APIs: {', '.join(detected_apis) if detected_apis else 'None (Basic Python)'}
2. Do NOT use any other APIs or create placeholder implementations
3. Include comprehensive error handling
4. Add rate limiting considerations if using APIs
5. Return structured data with clear success/error status
6. Include appropriate usage disclaimers

CRITICAL FUNCTION DESIGN REQUIREMENTS:
1. Functions must be fully self-contained and runnable without manual code modifications and unnecessary parameters
2. If using APIs, all API keys must be retrieved using os.getenv() INSIDE the function
3. NO function parameters for API keys or configuration
4. Use ONLY these verified APIs: {', '.join(detected_apis) if detected_apis else 'None (Basic Python)'}
4. Functions should have minimal or no parameters (only essential business logic parameters)
5. All required data (API keys, credentials) must be auto-injected from environment variables
6. It should only have parameters specified in user_request. Example: user_request = "Calculate factorial of a number" 
then generated function definition should be like "def calculate_factorial(n)" Here n is parameter from user_request
7. The function must be immediately executable without any code modifications

CRITICAL PYTHON SYNTAX REQUIREMENTS:
1. Use 4 spaces for indentation (NO tabs)
2. Maintain consistent indentation throughout the code
3. Follow PEP 8 style guidelines
4. Properly indent all code blocks (if, for, while, try, etc.)
5. Properly indent function definitions and their contents
6. Properly indent class definitions and their methods
7. Ensure all code blocks are properly closed
8. Use proper line breaks between functions and classes
9. Ensure all parentheses, brackets, and braces are properly closed
10. Use proper spacing around operators and after commas

CRITICAL ANTI-SIMULATION REQUIREMENTS:
1. NEVER create simulated/fake/placeholder data
2. NEVER use hardcoded values as substitutes for real API data
3. If user-specific data is needed, make it a function parameter
4. If sensitive data is needed, retrieve it from environment variables
5. If optional configuration is needed, provide it as a parameter with default value
6. NEVER calculate fake values or simulate responses
7. NEVER return mock data when real API calls fail
8. If you cannot get real data, return an error message - DO NOT simulate

{api_info if detected_apis else "This is a basic Python implementation that does not require external APIs."}

Create a single, complete function that:
- Is fully self-contained and requires NO external parameters for API keys and NO manual code modifications
- If using APIs, retrieves all API keys using os.getenv() inside the function
- Handles all specified requirements within API limitations (if any)
- Includes proper input validation
- Has comprehensive error handling
- Is syntactically correct Python code with proper indentation
- Returns meaningful Real results or clear error messages
- Includes rate limiting awareness if using APIs
- Has detailed docstring with limitations
- Has NO hardcoded values requiring manual replacement
- NEVER simulates or creates fake data
- Can be executed immediately with only required parameter passing

Do NOT:
- Create placeholder implementations for missing APIs
- Promise capabilities beyond available APIs
- Use external APIs not listed above
- Make unrealistic accuracy claims
- Simulate ANY data - always use real API responses or proper Python implementations
- Use inconsistent indentation

Format as complete Python code with imports and example usage.
"""
        
        try:
            messages = [
                SystemMessage(content="You are a code generator expert. Generate code that strictly follows the requirements, syntactically correct, follows PEP 8 style guidelines, uses proper indentation and never simulates data."),
                HumanMessage(content=prompt)
            ]
            
            response = self.model.invoke(messages)
            generated_code = response.content
            
            # Clean up the code
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].strip()
            
            # Clean up indentation
            generated_code = self._cleanup_code_indentation(generated_code)
            
            # Inject API key retrieval inside functions
            generated_code = self._inject_api_key_retrieval(generated_code, detected_apis)

            # POST-GENERATION VALIDATION: Check for simulation patterns
            simulation_patterns = [
                'simulated_price', 'fake_price', 'mock_price', 'placeholder',
                '100.0 +', '* 0.5', 'mock_data', 'fake_data', 'dummy_data'
            ]
            
            for pattern in simulation_patterns:
                if pattern in generated_code.lower():
                    raise Exception(f"Generated code contains simulation pattern: '{pattern}'. Real API implementation required.")
            
            return generated_code, detected_apis
            
        except Exception as e:
            raise Exception(f"Error generating code: {str(e)}")
    
    def _inject_api_key_retrieval(self, code: str, detected_apis: List[str]) -> str:
        """Ensure API keys are retrieved inside functions instead of passed as parameters."""
        
        # Remove api_key parameters from function definitions
        code = re.sub(r'def\s+(\w+)\s*\([^)]*api_key[^)]*\)', 
                    lambda m: m.group(0).replace('api_key, ', '').replace(', api_key', '').replace('api_key', ''), 
                    code)
        
        # Find function definitions and inject API key retrieval at the beginning
        for api in detected_apis:
            if api in self.available_apis and self.available_apis[api]['requires_key']:
                key_name = self.available_apis[api]['key_name']
                
                if isinstance(key_name, list):
                    # Multiple keys needed
                    key_retrievals = []
                    for key in key_name:
                        key_retrievals.append(f"    {key.lower().replace('_', '_')} = os.getenv('{key}')")
                        key_retrievals.append(f"    if not {key.lower().replace('_', '_')}:")
                        key_retrievals.append(f"        return {{'status': 'error', 'message': 'Missing {key} environment variable'}}")
                    key_injection = '\n'.join(key_retrievals)
                else:
                    # Single key needed
                    var_name = key_name.lower().replace('_key', '').replace('_', '_') + '_key'
                    key_injection = f"""    {var_name} = os.getenv('{key_name}')
        if not {var_name}:
            return {{'status': 'error', 'message': 'Missing {key_name} environment variable'}}"""
                
                # Inject after function definition
                pattern = r'(def\s+\w+\s*\([^)]*\):\s*\n\s*"""[^"]*"""\s*\n)'
                replacement = f'\\1\n{key_injection}\n'
                code = re.sub(pattern, replacement, code, flags=re.DOTALL)
        
        return code

    def standardize_api_keys(self, code: str) -> str:
        """Replace API key variable names with standardized ones."""
        for standard_name, actual_name in self.api_key_mapping.items():
            pattern = rf"os\.getenv\(['\"]({standard_name})['\"]"
            replacement = f"os.getenv('{actual_name}'"
            code = re.sub(pattern, replacement, code)
        return code
    
    def _cleanup_code_indentation(self, code: str) -> str:
        """Clean up code indentation to ensure it follows Python standards."""
        try:
            # Split code into lines
            lines = code.split('\n')
            cleaned_lines = []
            current_indent = 0
            in_multiline_string = False
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    cleaned_lines.append('')
                    continue
                
                # Handle multiline strings
                if '"""' in line or "'''" in line:
                    in_multiline_string = not in_multiline_string
                
                # Skip indentation fixes for lines in multiline strings
                if in_multiline_string:
                    cleaned_lines.append(line)
                    continue
                
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip())
                
                # Fix indentation to be multiple of 4
                if leading_spaces % 4 != 0:
                    # Round to nearest multiple of 4
                    leading_spaces = (leading_spaces // 4) * 4
                
                # Ensure consistent indentation
                if line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except ', 'else:', 'elif ')):
                    # Reset indentation for new blocks
                    current_indent = leading_spaces
                elif line.strip().startswith(('return', 'break', 'continue', 'pass')):
                    # These should be at the same level as their block
                    leading_spaces = current_indent
                elif line.strip().startswith(('else:', 'elif ', 'except ')):
                    # These should be at the same level as their corresponding if/try
                    leading_spaces = current_indent
                
                # Create properly indented line
                cleaned_line = ' ' * leading_spaces + line.lstrip()
                cleaned_lines.append(cleaned_line)
            
            # Join lines back together
            cleaned_code = '\n'.join(cleaned_lines)
            
            # Validate the cleaned code
            compile(cleaned_code, '<string>', 'exec')
            
            return cleaned_code
            
        except Exception as e:
            # If cleanup fails, return original code
            print(f"Warning: Code cleanup failed: {str(e)}")
            return code
        
    def _prompt_for_api_keys(self, missing_keys: List[str]) -> Dict[str, str]:
        """Prompt user for missing API keys."""
        provided_keys = {}
        print("\n🔑 Required API Keys:")
        for key in missing_keys:
            value = input(f"Enter {key}: ").strip()
            if value:
                provided_keys[key] = value
                # Optionally save to environment
                os.environ[key] = value
        return provided_keys

    def _suggest_api_sources(self, api_name: str) -> str:
        """Provide information about where to get specific API keys."""
        api_sources = {
            'weather': {
                'name': 'OpenWeatherMap API',
                'url': 'https://openweathermap.org/api',
                'description': 'Provides current weather data, forecasts, and historical weather data',
                'free_tier': '60 calls/minute, 1,000,000 calls/month',
                'key_name': 'OPENWEATHER_API_KEY'
            },
            'stock': {
                'name': 'Alpha Vantage API',
                'url': 'https://www.alphavantage.co/',
                'description': 'Provides real-time and historical stock data',
                'free_tier': '5 API calls per minute, 500 calls per day',
                'key_name': 'ALPHA_VANTAGE_API_KEY'
            },
            'news': {
                'name': 'NewsAPI',
                'url': 'https://newsapi.org/',
                'description': 'Provides news headlines and articles',
                'free_tier': '100 requests/day, 1 month history max',
                'key_name': 'NEWSAPI_KEY'
            }
        }
        
        if api_name.lower() in api_sources:
            source = api_sources[api_name.lower()]
            return f"""
📡 {source['name']}:
- Description: {source['description']}
- Free Tier: {source['free_tier']}
- Sign up at: {source['url']}
- Environment variable name: {source['key_name']}
"""
        return f"Please search online for a suitable {api_name} API provider."
    
    def save_tool(self, code: str, function_name: str, description: str, apis_used: List[str]) -> str:
        """Save the generated tool with enhanced headers."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{function_name}_{timestamp}.py"
        filepath = os.path.join(self.tools_directory, filename)
        
        standardized_code = self.standardize_api_keys(code)
        
        header = f'''"""
Generated Tool: {function_name}
Description: {description}
APIs Used: {', '.join(apis_used)}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DISCLAIMER: This tool is for educational and informational purposes only.
- Generated tools should be tested thoroughly before production use
- Always verify data from reliable sources
- Respect API rate limits and terms of service
- Handle errors and edge cases appropriately
- Keep API keys and sensitive data secure

API Limitations:
{chr(10).join([f"- {api}: {self.available_apis[api]['limitations']}" for api in apis_used if api in self.available_apis])}
"""

import os
from dotenv import load_dotenv
load_dotenv()

'''
        #saves in dynamic_tools directory
        with open(filepath, 'w') as f:
            f.write(header + standardized_code)
        
        #saves in tool_registry.json 
        self._update_registry(function_name, filename, description, apis_used)
        # Debug: Verify registry was updated
        try:
            with open(self.registry_file, 'r') as f:
                registry = json.load(f)
                print(f"📊 Registry now contains {len(registry.get('tools', []))} tools")
        except Exception as e:
            print(f"⚠️  Could not verify registry update: {e}")
        
        return filepath
    
    def _update_registry(self, function_name: str, filename: str, description: str, apis_used: List[str]):
        """Update the tool registry."""
        try:
            if not os.path.exists(self.registry_file):
                registry = {"tools": []}
            else:
                with open(self.registry_file, 'r') as f:
                    try:
                        registry = json.load(f)
                        if "tools" not in registry:
                            registry["tools"] = []
                    except json.JSONDecodeError:
                        registry = {"tools": []}
            
            tool_info = {
                "name": function_name,
                "filename": filename,
                "description": description,
                "apis_used": apis_used,
                "created_at": datetime.now().isoformat(),
                "filepath": os.path.join(self.tools_directory, filename)
            }
            
            #filepath = os.path.join(self.tools_directory, filename)
            registry["tools"].append(tool_info)
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
                
            print(f"✅ Tool registered successfully in {self.registry_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to update registry: {str(e)}")
            print(f"Registry file: {self.registry_file}")
            print(f"Current directory: {os.getcwd()}")
    
    def extract_function_name(self, code: str) -> str:
        """Extract the main function name from generated code."""
        function_matches = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        if function_matches:
            return function_matches[0]
        return "unknown_function"
    
    def _show_usage_reminders(self):
        """Show important usage reminders."""
        print("\n⚠️  IMPORTANT REMINDERS:")
        print("- This tool is for educational and development purposes only")
        print("- Generated tools should be tested thoroughly before production use")
        print("- Always verify data from reliable sources")
        print("- Respect API rate limits and terms of service")
        print("- Handle errors and edge cases appropriately")
        print("- Keep API keys and sensitive data secure")
        print("- Consider performance implications for large datasets")
        print("- Document any limitations or assumptions in your code")
    
    def list_available_apis(self):
        """Display comprehensive API information."""
        print("\n📡 Available APIs:")
        print("-" * 70)
        for api_name, api_info in self.available_apis.items():
            status = "✅" if not api_info['requires_key'] else "🔑"
            
            print(f"{status} {api_name}: {api_info['description']}")
            
            if api_info['requires_key']:
                if isinstance(api_info['key_name'], list):
                    all_keys_available = all(os.getenv(key) for key in api_info['key_name'])
                    key_available = "✅" if all_keys_available else "❌"
                    key_names = ', '.join(api_info['key_name'])
                    print(f"   Keys: {key_names} - {key_available}")
                else:
                    key_available = "✅" if os.getenv(api_info['key_name']) else "❌"
                    print(f"   Key: {api_info['key_name']} - {key_available}")
            
            print(f"   Capabilities: {', '.join(api_info['capabilities'])}")
            print(f"   Cannot do: {', '.join(api_info['cannot_do'])}")
            print(f"   Limitations: {api_info['limitations']}")
            print()

def main():
    """Main function with enhanced user guidance."""
    try:
        agent = CodeGeneratorAgent()
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    print("🤖 Robust Code Generator Agent Started!")
    print("=" * 70)
    print("IMPORTANT DISCLAIMERS:")
    print("- This tool creates educational tools only")
    print("- All required API keys must be configured before code generation")
    print("- Free APIs have strict limitations - no premium data access")
    print("- Generated tools are for learning purposes, not production use")
    print("=" * 70)
    print("Commands: 'exit' to quit, 'apis' to see available APIs, 'help' for guidelines")
    
    while True:
        user_input = input("\n💬 Describe the tool you want to create: ").strip()
        
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'apis':
            agent.list_available_apis()
            continue
        elif user_input.lower() == 'help':
            print("\n📖 USAGE GUIDELINES:")
            print("\n✅ REQUESTS THAT WILL BE ACCEPTED:")
            print("- 'Create a tool that fetches weather data for a city nagpur,india' (uses requests, openweather)") #❌ code with unicode error is generated
            print("- 'Build a news summarizer for a topic' (uses newsapi)") #✅
            print("- 'Generate a text analysis tool' (uses google_genai)") #✅
            print("I want a tool that generates daily reports from log files and emails them to user@gmail.com (uses email_smtp)") #✅
            print("Create a function that sends bulk SMS messages using Twilio API with personalized content") #✅
            #API Data Retrieval
            print("Create a tool that fetches cryptocurrency prices from CoinGecko API and returns them in a formatted dictionary") #❌/✅ using requests approach or API key approach
            print("Generate a tool that retrieves social media posts from Twitter API using hashtag search") #✅ needs new twitter api
            print("I need a function that gets the latest news headlines from a news API based on a specific topic or keyword") #✅ no addition api needed
            print("Create a function that fetches restaurant data from Yelp API for a given location and cuisine type")#✅ needs yelp api
            #Uses Basic Python (NO APIs)
            print("-Create a function that counts the frequency of words in a text file") #✅
            print("-Create a function that converts a text file to uppercase") #✅
            print("create a tool to find factorial of a number") #✅
            print("I want a tool that extracts email addresses from a given text string")#✅
            print("I need a function that compresses multiple files into a ZIP archive") #✅
            #Web Scraping
            print("I need a function that extracts all hyperlinks from a webpage and saves them to a text file")#✅
            print("Create a function that fetches restaurant data from Yelp API for a given location and cuisine type") #✅
            #API Data Processing & Transformation
            print("Create a tool that fetches weather data from OpenWeatherMap API and converts it to a CSV report") #✅
            print("I want a tool that gets stock market data from Alpha Vantage API and calculates moving averages") #✅ compound word detetction (2 words alpha vantage)
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: Insufficient detail")
            print("- 'Create a simple tool'") #✅
            print("- 'Get information'") #✅
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: Security Violation")
            print("- Build a function that hacks into systems") #✅
            print("- Access unauthorized data") #✅ insufficient detail + security violation
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: Impossible Request")
            print("- Create a tool that predicts with 100% accuracy") #✅
            print("- 'Generate code that creates infinite resources'") #✅
            print("- 'Make a tool that guarantees success'") #✅
            print("- 'Create a tool that reads my mind'") #✅
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: Premium/Paid Services")
            print("Create a tool that uploads files to cloud storage via API and returns shareable links") #requires interaction with cloud storage services
            
            print("\n❌ REQUESTS THAT WILL BE REJECTED: Data Heavy")
            print("- 'Create a tool that analyzes millions of records'") #✅
            print("- 'Build a function that processes terabytes of data'") #✅
            
            print("\n🔑 API KEY REQUIREMENTS:")
            print("- ALL required API keys must be configured before code generation")
            print("- No placeholder implementations for missing APIs")
            print("- Check 'apis' command to see which keys are missing")
            continue
        elif not user_input:
            continue
        
        result = agent.create_tool(user_input)
        print(f"\n📋 Result: {result['message']}")
        
        if result['status'] == 'rejected' and 'detected_apis' in result:
            detected = result['detected_apis']
            if detected:
                print(f"💡 APIs that would be needed: {', '.join(detected)}")

if __name__ == "__main__":
    main()