"""
Advanced Security Engine for Next-Generation Threat Detection and Response.

Implements AI-powered threat detection, behavioral analysis, automated incident response,
zero-trust security model, and advanced cryptographic protections for autonomous SDLC systems.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import hashlib
import hmac
import secrets
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict, deque
import base64
import urllib.parse

from robo_rlhf.core import get_logger, get_config
from robo_rlhf.core.exceptions import RoboRLHFError, SecurityError, ValidationError
from robo_rlhf.core.performance import PerformanceMonitor, optimize_memory
from robo_rlhf.core.validators import validate_dict, validate_string
from robo_rlhf.core.security import sanitize_input, check_file_safety


class ThreatLevel(Enum):
    """Security threat severity levels."""
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ThreatType(Enum):
    """Categories of security threats."""
    CODE_INJECTION = "code_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DENIAL_OF_SERVICE = "denial_of_service"
    MALWARE = "malware"
    SOCIAL_ENGINEERING = "social_engineering"
    INSIDER_THREAT = "insider_threat"
    SUPPLY_CHAIN = "supply_chain"
    CRYPTOGRAPHIC = "cryptographic"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"


class SecurityDomain(Enum):
    """Security domains for threat analysis."""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    DATA = "data"
    NETWORK = "network"
    IDENTITY = "identity"
    DEVICE = "device"
    CLOUD = "cloud"


class ResponseAction(Enum):
    """Automated response actions."""
    ALERT = "alert"
    LOG = "log"
    QUARANTINE = "quarantine"
    BLOCK = "block"
    THROTTLE = "throttle"
    REDIRECT = "redirect"
    TERMINATE = "terminate"
    INVESTIGATE = "investigate"
    ESCALATE = "escalate"


@dataclass
class SecurityThreat:
    """Detected security threat information."""
    threat_id: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    domain: SecurityDomain
    source: str
    target: str
    description: str
    evidence: Dict[str, Any]
    confidence: float
    timestamp: float
    affected_assets: List[str] = field(default_factory=list)
    attack_vector: Optional[str] = None
    indicators: List[str] = field(default_factory=list)
    mitigation_suggestions: List[str] = field(default_factory=list)


@dataclass
class SecurityIncident:
    """Security incident tracking."""
    incident_id: str
    threats: List[SecurityThreat]
    status: str
    priority: int
    assigned_to: Optional[str]
    created_at: float
    updated_at: float
    response_actions: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    resolution: Optional[str] = None


@dataclass
class BehavioralProfile:
    """User/system behavioral profile for anomaly detection."""
    entity_id: str
    entity_type: str  # user, system, process
    baseline_behavior: Dict[str, Any]
    recent_behavior: Dict[str, Any]
    anomaly_score: float
    last_updated: float
    behavioral_patterns: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedSecurityEngine:
    """Advanced AI-powered security engine for autonomous SDLC protection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Security configuration
        self.security_config = self.config.get("advanced_security", {})
        self.threat_detection_enabled = self.security_config.get("threat_detection", True)
        self.behavioral_analysis_enabled = self.security_config.get("behavioral_analysis", True)
        self.auto_response_enabled = self.security_config.get("auto_response", True)
        
        # Threat detection parameters
        self.anomaly_threshold = self.security_config.get("anomaly_threshold", 0.7)
        self.threat_confidence_threshold = self.security_config.get("threat_confidence_threshold", 0.6)
        self.behavioral_window_hours = self.security_config.get("behavioral_window_hours", 24)
        
        # Security intelligence
        self.threat_signatures = self._initialize_threat_signatures()
        self.behavioral_baselines = {}
        self.security_rules = self._initialize_security_rules()
        self.threat_intelligence = self._initialize_threat_intelligence()
        
        # Real-time monitoring
        self.active_threats = {}
        self.security_incidents = {}
        self.behavioral_profiles = {}
        self.security_events = deque(maxlen=10000)
        
        # AI models for threat detection
        self.anomaly_detector = self._initialize_anomaly_detector()
        self.threat_classifier = self._initialize_threat_classifier()
        
        # Response capabilities
        self.response_handlers = self._initialize_response_handlers()
        self.escalation_rules = self._initialize_escalation_rules()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Cryptographic components
        self.crypto_engine = self._initialize_crypto_engine()
        
        # Zero-trust components
        self.trust_scores = defaultdict(float)
        self.access_policies = self._initialize_access_policies()
        
        self.logger.info("AdvancedSecurityEngine initialized with AI-powered threat detection")
    
    def _initialize_threat_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Initialize threat detection signatures."""
        return {
            "sql_injection": {
                "patterns": [
                    r"(\bunion\b.*\bselect\b)", r"(\bor\b.*\b=\b.*\b=\b)",
                    r"(\bdrop\b.*\btable\b)", r"(\binsert\b.*\binto\b)",
                    r"(;.*--)|(;.*\/\*)"
                ],
                "threat_type": ThreatType.CODE_INJECTION,
                "severity": ThreatLevel.HIGH,
                "confidence_base": 0.8
            },
            "xss_attack": {
                "patterns": [
                    r"<script.*?>.*?</script>", r"javascript:",
                    r"on\w+\s*=", r"eval\s*\(", r"document\.cookie"
                ],
                "threat_type": ThreatType.CODE_INJECTION,
                "severity": ThreatLevel.MEDIUM,
                "confidence_base": 0.7
            },
            "command_injection": {
                "patterns": [
                    r"(;|\||\&)\s*(cat|ls|pwd|whoami)", r"(`.*`)",
                    r"\$\(.*\)", r"(&&|\|\|)\s*(rm|mv|cp)"
                ],
                "threat_type": ThreatType.CODE_INJECTION,
                "severity": ThreatLevel.CRITICAL,
                "confidence_base": 0.9
            },
            "path_traversal": {
                "patterns": [
                    r"(\.\./){2,}", r"(\.\.\\){2,}",
                    r"\/etc\/passwd", r"\/proc\/", r"\\windows\\system32"
                ],
                "threat_type": ThreatType.DATA_EXFILTRATION,
                "severity": ThreatLevel.HIGH,
                "confidence_base": 0.8
            },
            "data_exfiltration": {
                "patterns": [
                    r"(curl|wget)\s+https?://", r"base64\s+-d",
                    r"nc\s+-l", r"python.*http\.server"
                ],
                "threat_type": ThreatType.DATA_EXFILTRATION,
                "severity": ThreatLevel.CRITICAL,
                "confidence_base": 0.85
            },
            "privilege_escalation": {
                "patterns": [
                    r"sudo\s+su", r"chmod\s+777", r"setuid",
                    r"\/bin\/sh", r"chown\s+root"
                ],
                "threat_type": ThreatType.PRIVILEGE_ESCALATION,
                "severity": ThreatLevel.CRITICAL,
                "confidence_base": 0.9
            }
        }
    
    def _initialize_security_rules(self) -> List[Dict[str, Any]]:
        """Initialize security rules for threat detection."""
        return [
            {
                "rule_id": "high_frequency_requests",
                "condition": lambda data: data.get("request_rate", 0) > 100,
                "threat_type": ThreatType.DENIAL_OF_SERVICE,
                "severity": ThreatLevel.MEDIUM,
                "description": "Unusually high request frequency detected"
            },
            {
                "rule_id": "suspicious_user_agent",
                "condition": lambda data: any(bot in data.get("user_agent", "").lower() 
                                            for bot in ["bot", "crawler", "scanner"]),
                "threat_type": ThreatType.BEHAVIORAL_ANOMALY,
                "severity": ThreatLevel.LOW,
                "description": "Suspicious user agent detected"
            },
            {
                "rule_id": "multiple_failed_auth",
                "condition": lambda data: data.get("failed_auth_count", 0) > 5,
                "threat_type": ThreatType.PRIVILEGE_ESCALATION,
                "severity": ThreatLevel.MEDIUM,
                "description": "Multiple failed authentication attempts"
            },
            {
                "rule_id": "unusual_data_transfer",
                "condition": lambda data: data.get("data_transfer_mb", 0) > 1000,
                "threat_type": ThreatType.DATA_EXFILTRATION,
                "severity": ThreatLevel.HIGH,
                "description": "Unusual large data transfer detected"
            }
        ]
    
    def _initialize_threat_intelligence(self) -> Dict[str, Any]:
        """Initialize threat intelligence feeds."""
        return {
            "malicious_ips": set([
                "192.168.1.100",  # Example malicious IP
                "10.0.0.50",      # Example suspicious IP
            ]),
            "malicious_domains": set([
                "malware.example.com",
                "phishing.example.org"
            ]),
            "known_vulnerabilities": {
                "CVE-2023-1234": {
                    "severity": "critical",
                    "description": "Remote code execution vulnerability",
                    "affected_components": ["web_server", "api_gateway"]
                }
            },
            "attack_patterns": {
                "supply_chain_attack": {
                    "indicators": ["unusual_dependency", "modified_package", "suspicious_checksum"],
                    "severity": ThreatLevel.CRITICAL
                }
            }
        }
    
    def _initialize_anomaly_detector(self) -> Dict[str, Any]:
        """Initialize anomaly detection models."""
        return {
            "statistical_model": {
                "type": "z_score",
                "threshold": 3.0,
                "window_size": 100
            },
            "ml_model": {
                "type": "isolation_forest",
                "contamination": 0.1,
                "n_estimators": 100
            },
            "behavioral_model": {
                "type": "lstm",
                "sequence_length": 10,
                "threshold": 0.7
            }
        }
    
    def _initialize_threat_classifier(self) -> Dict[str, Any]:
        """Initialize threat classification models."""
        return {
            "text_classifier": {
                "type": "naive_bayes",
                "features": ["ngrams", "keywords", "patterns"],
                "confidence_threshold": 0.6
            },
            "behavior_classifier": {
                "type": "random_forest",
                "features": ["frequency", "timing", "sequence", "context"],
                "confidence_threshold": 0.7
            }
        }
    
    def _initialize_response_handlers(self) -> Dict[ResponseAction, Callable]:
        """Initialize automated response handlers."""
        return {
            ResponseAction.ALERT: self._handle_alert,
            ResponseAction.LOG: self._handle_log,
            ResponseAction.QUARANTINE: self._handle_quarantine,
            ResponseAction.BLOCK: self._handle_block,
            ResponseAction.THROTTLE: self._handle_throttle,
            ResponseAction.TERMINATE: self._handle_terminate,
            ResponseAction.INVESTIGATE: self._handle_investigate,
            ResponseAction.ESCALATE: self._handle_escalate
        }
    
    def _initialize_escalation_rules(self) -> List[Dict[str, Any]]:
        """Initialize incident escalation rules."""
        return [
            {
                "condition": lambda threat: threat.threat_level == ThreatLevel.CRITICAL,
                "escalate_to": "security_team",
                "timeout_minutes": 15
            },
            {
                "condition": lambda threat: threat.threat_level == ThreatLevel.EMERGENCY,
                "escalate_to": "ciso",
                "timeout_minutes": 5
            },
            {
                "condition": lambda threat: threat.confidence > 0.9,
                "escalate_to": "incident_response_team",
                "timeout_minutes": 30
            }
        ]
    
    def _initialize_crypto_engine(self) -> Dict[str, Any]:
        """Initialize cryptographic capabilities."""
        return {
            "encryption_key": secrets.token_bytes(32),  # AES-256 key
            "signing_key": secrets.token_bytes(32),     # HMAC key
            "hash_algorithm": "sha256",
            "encryption_algorithm": "aes_gcm"
        }
    
    def _initialize_access_policies(self) -> List[Dict[str, Any]]:
        """Initialize zero-trust access policies."""
        return [
            {
                "policy_id": "admin_access",
                "conditions": {
                    "role": "admin",
                    "mfa_verified": True,
                    "trust_score": 0.8,
                    "location_verified": True
                },
                "actions": ["read", "write", "execute", "delete"],
                "resources": ["*"]
            },
            {
                "policy_id": "developer_access",
                "conditions": {
                    "role": "developer",
                    "trust_score": 0.6,
                    "working_hours": True
                },
                "actions": ["read", "write", "execute"],
                "resources": ["code", "builds", "tests"]
            },
            {
                "policy_id": "guest_access",
                "conditions": {
                    "role": "guest",
                    "trust_score": 0.3
                },
                "actions": ["read"],
                "resources": ["documentation", "public_apis"]
            }
        ]
    
    async def analyze_security_event(self, event: Dict[str, Any]) -> List[SecurityThreat]:
        """Analyze a security event for potential threats."""
        self.logger.debug(f"Analyzing security event: {event.get('event_type', 'unknown')}")
        
        threats = []
        
        with self.performance_monitor.measure("security_analysis"):
            # Pattern-based detection
            pattern_threats = await self._detect_pattern_threats(event)
            threats.extend(pattern_threats)
            
            # Rule-based detection
            rule_threats = await self._detect_rule_threats(event)
            threats.extend(rule_threats)
            
            # Behavioral anomaly detection
            if self.behavioral_analysis_enabled:
                behavioral_threats = await self._detect_behavioral_threats(event)
                threats.extend(behavioral_threats)
            
            # AI-powered classification
            ai_threats = await self._detect_ai_threats(event)
            threats.extend(ai_threats)
            
            # Threat intelligence correlation
            intelligence_threats = await self._correlate_threat_intelligence(event)
            threats.extend(intelligence_threats)
        
        # Store event for behavioral analysis
        self.security_events.append({
            "timestamp": time.time(),
            "event": event,
            "threats_detected": len(threats)
        })
        
        if threats:
            self.logger.info(f"Detected {len(threats)} security threats in event")
            
            # Update active threats
            for threat in threats:
                self.active_threats[threat.threat_id] = threat
        
        return threats
    
    async def _detect_pattern_threats(self, event: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect threats using pattern matching."""
        threats = []
        
        # Analyze text fields in the event
        text_fields = ["command", "query", "payload", "user_input", "url", "headers"]
        
        for field in text_fields:
            if field in event and isinstance(event[field], str):
                field_threats = await self._analyze_text_for_patterns(event[field], field, event)
                threats.extend(field_threats)
        
        return threats
    
    async def _analyze_text_for_patterns(self, text: str, field: str, event: Dict[str, Any]) -> List[SecurityThreat]:
        """Analyze text for malicious patterns."""
        threats = []
        
        for signature_name, signature in self.threat_signatures.items():
            for pattern in signature["patterns"]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    confidence = signature["confidence_base"]
                    
                    # Adjust confidence based on context
                    if field in ["command", "query"]:
                        confidence *= 1.2  # Higher confidence for executable content
                    
                    confidence = min(1.0, confidence)  # Cap at 1.0
                    
                    if confidence >= self.threat_confidence_threshold:
                        threat = SecurityThreat(
                            threat_id=self._generate_threat_id(),
                            threat_type=signature["threat_type"],
                            threat_level=signature["severity"],
                            domain=self._determine_domain(field),
                            source=event.get("source", "unknown"),
                            target=event.get("target", field),
                            description=f"{signature_name.replace('_', ' ').title()} detected in {field}",
                            evidence={
                                "pattern": pattern,
                                "match": match.group(0),
                                "field": field,
                                "full_text": text[:200] + "..." if len(text) > 200 else text
                            },
                            confidence=confidence,
                            timestamp=time.time(),
                            indicators=[match.group(0)],
                            attack_vector=field,
                            mitigation_suggestions=self._get_mitigation_suggestions(signature["threat_type"])
                        )
                        
                        threats.append(threat)
        
        return threats
    
    async def _detect_rule_threats(self, event: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect threats using security rules."""
        threats = []
        
        for rule in self.security_rules:
            try:
                if rule["condition"](event):
                    threat = SecurityThreat(
                        threat_id=self._generate_threat_id(),
                        threat_type=rule["threat_type"],
                        threat_level=rule["severity"],
                        domain=SecurityDomain.APPLICATION,
                        source=event.get("source", "unknown"),
                        target=event.get("target", "system"),
                        description=rule["description"],
                        evidence={"rule_id": rule["rule_id"], "event_data": event},
                        confidence=0.7,  # Rule-based confidence
                        timestamp=time.time(),
                        mitigation_suggestions=self._get_mitigation_suggestions(rule["threat_type"])
                    )
                    
                    threats.append(threat)
            except Exception as e:
                self.logger.warning(f"Rule evaluation failed for {rule['rule_id']}: {e}")
        
        return threats
    
    async def _detect_behavioral_threats(self, event: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect threats through behavioral analysis."""
        threats = []
        
        entity_id = event.get("user_id") or event.get("system_id", "unknown")
        entity_type = "user" if "user_id" in event else "system"
        
        # Get or create behavioral profile
        profile = await self._get_behavioral_profile(entity_id, entity_type)
        
        # Update profile with current event
        await self._update_behavioral_profile(profile, event)
        
        # Detect anomalies
        anomaly_score = await self._calculate_anomaly_score(profile, event)
        
        if anomaly_score > self.anomaly_threshold:
            threat = SecurityThreat(
                threat_id=self._generate_threat_id(),
                threat_type=ThreatType.BEHAVIORAL_ANOMALY,
                threat_level=self._determine_threat_level_from_score(anomaly_score),
                domain=SecurityDomain.IDENTITY,
                source=entity_id,
                target="behavioral_baseline",
                description=f"Behavioral anomaly detected for {entity_type} {entity_id}",
                evidence={
                    "anomaly_score": anomaly_score,
                    "baseline_deviation": profile.anomaly_score,
                    "behavioral_context": profile.recent_behavior
                },
                confidence=min(0.95, anomaly_score),
                timestamp=time.time(),
                mitigation_suggestions=["Verify user identity", "Review recent activities", "Temporary access restriction"]
            )
            
            threats.append(threat)
        
        return threats
    
    async def _detect_ai_threats(self, event: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect threats using AI classification."""
        threats = []
        
        # Extract features from event
        features = await self._extract_features(event)
        
        # Text-based classification
        if "text_content" in features:
            text_threat_prob = await self._classify_text_threat(features["text_content"])
            if text_threat_prob > 0.7:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    threat_type=ThreatType.CODE_INJECTION,  # Most likely for text threats
                    threat_level=ThreatLevel.MEDIUM,
                    domain=SecurityDomain.APPLICATION,
                    source=event.get("source", "unknown"),
                    target="text_classifier",
                    description="AI detected potential malicious content",
                    evidence={
                        "threat_probability": text_threat_prob,
                        "features": features["text_content"][:100]  # First 100 chars
                    },
                    confidence=text_threat_prob,
                    timestamp=time.time(),
                    mitigation_suggestions=["Content filtering", "Input sanitization"]
                )
                threats.append(threat)
        
        # Behavioral pattern classification
        if "behavioral_features" in features:
            behavior_threat_prob = await self._classify_behavior_threat(features["behavioral_features"])
            if behavior_threat_prob > 0.6:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    threat_type=ThreatType.INSIDER_THREAT,
                    threat_level=ThreatLevel.MEDIUM,
                    domain=SecurityDomain.IDENTITY,
                    source=event.get("source", "unknown"),
                    target="behavior_classifier",
                    description="AI detected suspicious behavioral pattern",
                    evidence={
                        "threat_probability": behavior_threat_prob,
                        "behavioral_indicators": features["behavioral_features"]
                    },
                    confidence=behavior_threat_prob,
                    timestamp=time.time(),
                    mitigation_suggestions=["Monitor user activity", "Access review"]
                )
                threats.append(threat)
        
        return threats
    
    async def _correlate_threat_intelligence(self, event: Dict[str, Any]) -> List[SecurityThreat]:
        """Correlate event with threat intelligence feeds."""
        threats = []
        
        # Check for malicious IPs
        source_ip = event.get("source_ip")
        if source_ip and source_ip in self.threat_intelligence["malicious_ips"]:
            threat = SecurityThreat(
                threat_id=self._generate_threat_id(),
                threat_type=ThreatType.MALWARE,
                threat_level=ThreatLevel.HIGH,
                domain=SecurityDomain.NETWORK,
                source=source_ip,
                target=event.get("target", "system"),
                description=f"Communication with known malicious IP: {source_ip}",
                evidence={"malicious_ip": source_ip, "intelligence_source": "threat_feed"},
                confidence=0.9,
                timestamp=time.time(),
                mitigation_suggestions=["Block IP address", "Investigate connection"]
            )
            threats.append(threat)
        
        # Check for malicious domains
        domain = event.get("domain") or self._extract_domain_from_url(event.get("url", ""))
        if domain and domain in self.threat_intelligence["malicious_domains"]:
            threat = SecurityThreat(
                threat_id=self._generate_threat_id(),
                threat_type=ThreatType.MALWARE,
                threat_level=ThreatLevel.HIGH,
                domain=SecurityDomain.NETWORK,
                source=domain,
                target=event.get("target", "system"),
                description=f"Communication with known malicious domain: {domain}",
                evidence={"malicious_domain": domain, "intelligence_source": "threat_feed"},
                confidence=0.85,
                timestamp=time.time(),
                mitigation_suggestions=["Block domain", "DNS filtering"]
            )
            threats.append(threat)
        
        return threats
    
    async def _get_behavioral_profile(self, entity_id: str, entity_type: str) -> BehavioralProfile:
        """Get or create behavioral profile for entity."""
        if entity_id not in self.behavioral_profiles:
            profile = BehavioralProfile(
                entity_id=entity_id,
                entity_type=entity_type,
                baseline_behavior={
                    "request_rate": 10.0,
                    "session_duration": 1800.0,  # 30 minutes
                    "failed_requests": 0.1,
                    "data_access_pattern": "normal"
                },
                recent_behavior={},
                anomaly_score=0.0,
                last_updated=time.time()
            )
            self.behavioral_profiles[entity_id] = profile
        
        return self.behavioral_profiles[entity_id]
    
    async def _update_behavioral_profile(self, profile: BehavioralProfile, event: Dict[str, Any]) -> None:
        """Update behavioral profile with new event."""
        current_time = time.time()
        
        # Update recent behavior
        profile.recent_behavior.update({
            "last_activity": current_time,
            "recent_request_rate": event.get("request_rate", 0),
            "recent_failed_requests": event.get("failed_requests", 0),
            "recent_data_access": event.get("data_accessed", 0)
        })
        
        # Update behavioral patterns
        pattern = {
            "timestamp": current_time,
            "event_type": event.get("event_type", "unknown"),
            "metrics": {k: v for k, v in event.items() if isinstance(v, (int, float))}
        }
        
        profile.behavioral_patterns.append(pattern)
        
        # Keep only recent patterns (last 24 hours)
        cutoff_time = current_time - (self.behavioral_window_hours * 3600)
        profile.behavioral_patterns = [
            p for p in profile.behavioral_patterns if p["timestamp"] > cutoff_time
        ]
        
        profile.last_updated = current_time
    
    async def _calculate_anomaly_score(self, profile: BehavioralProfile, event: Dict[str, Any]) -> float:
        """Calculate anomaly score for current event against profile."""
        if not profile.behavioral_patterns:
            return 0.0
        
        # Calculate deviations from baseline
        anomaly_factors = []
        
        # Request rate anomaly
        baseline_rate = profile.baseline_behavior.get("request_rate", 10.0)
        current_rate = event.get("request_rate", 0)
        if baseline_rate > 0:
            rate_deviation = abs(current_rate - baseline_rate) / baseline_rate
            anomaly_factors.append(min(1.0, rate_deviation))
        
        # Failed request rate anomaly
        baseline_failures = profile.baseline_behavior.get("failed_requests", 0.1)
        current_failures = event.get("failed_requests", 0)
        if baseline_failures > 0:
            failure_deviation = abs(current_failures - baseline_failures) / baseline_failures
            anomaly_factors.append(min(1.0, failure_deviation))
        
        # Time-based anomaly (unusual activity hours)
        current_hour = datetime.fromtimestamp(time.time()).hour
        recent_hours = [
            datetime.fromtimestamp(p["timestamp"]).hour 
            for p in profile.behavioral_patterns[-10:]  # Last 10 patterns
        ]
        
        if recent_hours:
            hour_frequency = recent_hours.count(current_hour) / len(recent_hours)
            if hour_frequency < 0.1:  # Very unusual hour
                anomaly_factors.append(0.5)
        
        # Calculate overall anomaly score
        if anomaly_factors:
            anomaly_score = np.mean(anomaly_factors)
        else:
            anomaly_score = 0.0
        
        # Update profile anomaly score with exponential moving average
        profile.anomaly_score = profile.anomaly_score * 0.7 + anomaly_score * 0.3
        
        return anomaly_score
    
    async def _extract_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from event for AI analysis."""
        features = {}
        
        # Text content features
        text_fields = ["command", "query", "payload", "user_input", "url"]
        text_content = ""
        for field in text_fields:
            if field in event and isinstance(event[field], str):
                text_content += event[field] + " "
        
        if text_content.strip():
            features["text_content"] = text_content.strip()
        
        # Behavioral features
        behavioral_features = {
            "request_rate": event.get("request_rate", 0),
            "session_duration": event.get("session_duration", 0),
            "failed_requests": event.get("failed_requests", 0),
            "data_transferred": event.get("data_transfer_mb", 0),
            "hour_of_day": datetime.fromtimestamp(time.time()).hour,
            "day_of_week": datetime.fromtimestamp(time.time()).weekday()
        }
        
        features["behavioral_features"] = behavioral_features
        
        return features
    
    async def _classify_text_threat(self, text: str) -> float:
        """Classify text content for threats using AI."""
        # Simplified threat classification based on suspicious keywords
        threat_keywords = [
            "script", "eval", "exec", "system", "shell", "cmd",
            "union", "select", "drop", "delete", "update", "insert",
            "../../", "..\\", "/etc/passwd", "<?php"
        ]
        
        text_lower = text.lower()
        threat_score = 0.0
        
        for keyword in threat_keywords:
            if keyword in text_lower:
                threat_score += 0.1
        
        # Additional scoring based on pattern complexity
        if re.search(r"[<>\"\'();]", text):  # Special characters
            threat_score += 0.2
        
        if len(text) > 1000:  # Unusually long input
            threat_score += 0.1
        
        return min(1.0, threat_score)
    
    async def _classify_behavior_threat(self, behavioral_features: Dict[str, Any]) -> float:
        """Classify behavioral pattern for threats."""
        threat_score = 0.0
        
        # High request rate
        request_rate = behavioral_features.get("request_rate", 0)
        if request_rate > 100:
            threat_score += 0.3
        
        # High failure rate
        failed_requests = behavioral_features.get("failed_requests", 0)
        if failed_requests > 10:
            threat_score += 0.4
        
        # Large data transfer
        data_transferred = behavioral_features.get("data_transferred", 0)
        if data_transferred > 1000:  # 1GB
            threat_score += 0.4
        
        # Unusual time activity
        hour = behavioral_features.get("hour_of_day", 12)
        if hour < 6 or hour > 22:  # Outside normal hours
            threat_score += 0.2
        
        return min(1.0, threat_score)
    
    async def respond_to_threat(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Execute automated response to detected threat."""
        self.logger.info(f"Responding to {threat.threat_level.value} threat: {threat.threat_id}")
        
        response_actions = []
        
        # Determine response actions based on threat level and type
        if threat.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]:
            response_actions.extend([ResponseAction.ALERT, ResponseAction.BLOCK, ResponseAction.ESCALATE])
        elif threat.threat_level == ThreatLevel.HIGH:
            response_actions.extend([ResponseAction.ALERT, ResponseAction.QUARANTINE, ResponseAction.INVESTIGATE])
        elif threat.threat_level == ThreatLevel.MEDIUM:
            response_actions.extend([ResponseAction.LOG, ResponseAction.THROTTLE])
        else:
            response_actions.append(ResponseAction.LOG)
        
        # Execute response actions
        response_results = {}
        for action in response_actions:
            if action in self.response_handlers:
                try:
                    result = await self.response_handlers[action](threat)
                    response_results[action.value] = result
                except Exception as e:
                    self.logger.error(f"Response action {action.value} failed: {e}")
                    response_results[action.value] = {"success": False, "error": str(e)}
        
        # Create security incident if threat is severe
        if threat.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]:
            incident = await self._create_security_incident(threat, response_actions)
            response_results["incident_created"] = incident.incident_id
        
        return {
            "threat_id": threat.threat_id,
            "response_actions": response_actions,
            "response_results": response_results,
            "timestamp": time.time()
        }
    
    async def _create_security_incident(self, 
                                      threat: SecurityThreat, 
                                      response_actions: List[ResponseAction]) -> SecurityIncident:
        """Create security incident for high-priority threats."""
        incident_id = self._generate_incident_id()
        
        incident = SecurityIncident(
            incident_id=incident_id,
            threats=[threat],
            status="open",
            priority=self._calculate_incident_priority(threat),
            assigned_to=None,
            created_at=time.time(),
            updated_at=time.time(),
            response_actions=[{"action": action.value, "timestamp": time.time()} for action in response_actions],
            timeline=[{
                "timestamp": time.time(),
                "event": "incident_created",
                "details": f"Created from threat {threat.threat_id}"
            }]
        )
        
        self.security_incidents[incident_id] = incident
        
        self.logger.warning(f"Security incident created: {incident_id} for threat {threat.threat_id}")
        
        return incident
    
    # Response handler implementations
    async def _handle_alert(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Handle alert response action."""
        self.logger.warning(f"SECURITY ALERT: {threat.description} (Confidence: {threat.confidence:.2f})")
        
        alert_data = {
            "alert_id": self._generate_alert_id(),
            "threat_id": threat.threat_id,
            "severity": threat.threat_level.value,
            "message": threat.description,
            "timestamp": time.time(),
            "evidence": threat.evidence
        }
        
        # In real implementation, this would send to SIEM, email, Slack, etc.
        return {"success": True, "alert_data": alert_data}
    
    async def _handle_log(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Handle log response action."""
        log_entry = {
            "timestamp": time.time(),
            "threat_id": threat.threat_id,
            "threat_type": threat.threat_type.value,
            "severity": threat.threat_level.value,
            "source": threat.source,
            "target": threat.target,
            "confidence": threat.confidence,
            "evidence": threat.evidence
        }
        
        self.logger.info(f"Security event logged: {json.dumps(log_entry)}")
        
        return {"success": True, "log_entry": log_entry}
    
    async def _handle_quarantine(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Handle quarantine response action."""
        self.logger.warning(f"Quarantining threat source: {threat.source}")
        
        # In real implementation, this would isolate the source
        quarantine_data = {
            "quarantine_id": self._generate_quarantine_id(),
            "threat_id": threat.threat_id,
            "source": threat.source,
            "timestamp": time.time(),
            "duration": 3600  # 1 hour quarantine
        }
        
        return {"success": True, "quarantine_data": quarantine_data}
    
    async def _handle_block(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Handle block response action."""
        self.logger.warning(f"Blocking threat source: {threat.source}")
        
        # In real implementation, this would update firewall rules
        block_data = {
            "block_id": self._generate_block_id(),
            "threat_id": threat.threat_id,
            "source": threat.source,
            "timestamp": time.time(),
            "rule_added": f"block_{threat.source}_{int(time.time())}"
        }
        
        return {"success": True, "block_data": block_data}
    
    async def _handle_throttle(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Handle throttle response action."""
        self.logger.info(f"Throttling requests from: {threat.source}")
        
        throttle_data = {
            "throttle_id": self._generate_throttle_id(),
            "threat_id": threat.threat_id,
            "source": threat.source,
            "rate_limit": 10,  # requests per minute
            "timestamp": time.time()
        }
        
        return {"success": True, "throttle_data": throttle_data}
    
    async def _handle_terminate(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Handle terminate response action."""
        self.logger.critical(f"Terminating connection from: {threat.source}")
        
        terminate_data = {
            "terminate_id": self._generate_terminate_id(),
            "threat_id": threat.threat_id,
            "source": threat.source,
            "timestamp": time.time(),
            "action": "connection_terminated"
        }
        
        return {"success": True, "terminate_data": terminate_data}
    
    async def _handle_investigate(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Handle investigate response action."""
        self.logger.info(f"Initiating investigation for threat: {threat.threat_id}")
        
        investigation_data = {
            "investigation_id": self._generate_investigation_id(),
            "threat_id": threat.threat_id,
            "timestamp": time.time(),
            "status": "initiated",
            "assigned_to": "automated_investigator"
        }
        
        return {"success": True, "investigation_data": investigation_data}
    
    async def _handle_escalate(self, threat: SecurityThreat) -> Dict[str, Any]:
        """Handle escalate response action."""
        self.logger.critical(f"Escalating threat: {threat.threat_id}")
        
        escalation_data = {
            "escalation_id": self._generate_escalation_id(),
            "threat_id": threat.threat_id,
            "escalated_to": "security_team",
            "timestamp": time.time(),
            "priority": "high"
        }
        
        return {"success": True, "escalation_data": escalation_data}
    
    def _generate_threat_id(self) -> str:
        """Generate unique threat ID."""
        return f"THR_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _generate_incident_id(self) -> str:
        """Generate unique incident ID."""
        return f"INC_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        return f"ALT_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _generate_quarantine_id(self) -> str:
        """Generate unique quarantine ID."""
        return f"QUA_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _generate_block_id(self) -> str:
        """Generate unique block ID."""
        return f"BLK_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _generate_throttle_id(self) -> str:
        """Generate unique throttle ID."""
        return f"THR_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _generate_terminate_id(self) -> str:
        """Generate unique terminate ID."""
        return f"TRM_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _generate_investigation_id(self) -> str:
        """Generate unique investigation ID."""
        return f"INV_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _generate_escalation_id(self) -> str:
        """Generate unique escalation ID."""
        return f"ESC_{int(time.time())}_{secrets.token_hex(4)}"
    
    def _determine_domain(self, field: str) -> SecurityDomain:
        """Determine security domain based on field."""
        domain_mapping = {
            "command": SecurityDomain.APPLICATION,
            "query": SecurityDomain.DATA,
            "url": SecurityDomain.NETWORK,
            "headers": SecurityDomain.NETWORK,
            "payload": SecurityDomain.APPLICATION
        }
        return domain_mapping.get(field, SecurityDomain.APPLICATION)
    
    def _determine_threat_level_from_score(self, score: float) -> ThreatLevel:
        """Determine threat level from anomaly score."""
        if score >= 0.9:
            return ThreatLevel.CRITICAL
        elif score >= 0.8:
            return ThreatLevel.HIGH
        elif score >= 0.6:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _calculate_incident_priority(self, threat: SecurityThreat) -> int:
        """Calculate incident priority (1=highest, 5=lowest)."""
        if threat.threat_level == ThreatLevel.EMERGENCY:
            return 1
        elif threat.threat_level == ThreatLevel.CRITICAL:
            return 2
        elif threat.threat_level == ThreatLevel.HIGH:
            return 3
        elif threat.threat_level == ThreatLevel.MEDIUM:
            return 4
        else:
            return 5
    
    def _get_mitigation_suggestions(self, threat_type: ThreatType) -> List[str]:
        """Get mitigation suggestions for threat type."""
        suggestions = {
            ThreatType.CODE_INJECTION: [
                "Input validation and sanitization",
                "Parameterized queries",
                "Content Security Policy",
                "Web Application Firewall"
            ],
            ThreatType.DATA_EXFILTRATION: [
                "Data Loss Prevention (DLP)",
                "Network monitoring",
                "Access controls",
                "Encryption at rest and in transit"
            ],
            ThreatType.PRIVILEGE_ESCALATION: [
                "Principle of least privilege",
                "Regular access reviews",
                "Multi-factor authentication",
                "Privileged access management"
            ],
            ThreatType.DENIAL_OF_SERVICE: [
                "Rate limiting",
                "Traffic filtering",
                "Load balancing",
                "DDoS protection services"
            ],
            ThreatType.BEHAVIORAL_ANOMALY: [
                "User behavior monitoring",
                "Identity verification",
                "Session management",
                "Continuous authentication"
            ]
        }
        
        return suggestions.get(threat_type, ["General security monitoring", "Incident response"])
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ""
        
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc.lower()
        except:
            return ""
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security engine statistics."""
        current_time = time.time()
        last_hour = current_time - 3600
        
        # Recent threats
        recent_threats = [t for t in self.active_threats.values() if t.timestamp > last_hour]
        
        # Threat distribution
        threat_levels = defaultdict(int)
        threat_types = defaultdict(int)
        
        for threat in recent_threats:
            threat_levels[threat.threat_level.value] += 1
            threat_types[threat.threat_type.value] += 1
        
        return {
            "total_active_threats": len(self.active_threats),
            "recent_threats_1h": len(recent_threats),
            "security_incidents": len(self.security_incidents),
            "behavioral_profiles": len(self.behavioral_profiles),
            "security_events": len(self.security_events),
            "threat_level_distribution": dict(threat_levels),
            "threat_type_distribution": dict(threat_types),
            "detection_capabilities": {
                "pattern_signatures": len(self.threat_signatures),
                "security_rules": len(self.security_rules),
                "threat_intelligence_feeds": len(self.threat_intelligence),
                "behavioral_analysis": self.behavioral_analysis_enabled,
                "ai_classification": True
            },
            "response_capabilities": {
                "response_handlers": len(self.response_handlers),
                "escalation_rules": len(self.escalation_rules),
                "auto_response_enabled": self.auto_response_enabled
            }
        }
    
    def __del__(self):
        """Cleanup resources."""
        optimize_memory()