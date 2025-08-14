"""
Global-first implementation with internationalization, compliance, and cross-platform support.

Multi-language support, regulatory compliance (GDPR, CCPA, PDPA), and cross-platform 
compatibility for worldwide quantum RLHF deployment.
"""

import os
import json
import locale
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading

from robo_rlhf.core import get_logger, get_config
from robo_rlhf.core.exceptions import RoboRLHFError, ValidationError


class Locale(Enum):
    """Supported locales for internationalization."""
    EN_US = "en_US"  # English (United States)
    EN_GB = "en_GB"  # English (United Kingdom)
    ES_ES = "es_ES"  # Spanish (Spain)
    ES_MX = "es_MX"  # Spanish (Mexico)
    FR_FR = "fr_FR"  # French (France)
    FR_CA = "fr_CA"  # French (Canada)
    DE_DE = "de_DE"  # German (Germany)
    DE_AT = "de_AT"  # German (Austria)
    IT_IT = "it_IT"  # Italian (Italy)
    PT_BR = "pt_BR"  # Portuguese (Brazil)
    PT_PT = "pt_PT"  # Portuguese (Portugal)
    RU_RU = "ru_RU"  # Russian (Russia)
    ZH_CN = "zh_CN"  # Chinese (Simplified)
    ZH_TW = "zh_TW"  # Chinese (Traditional)
    JA_JP = "ja_JP"  # Japanese (Japan)
    KO_KR = "ko_KR"  # Korean (South Korea)
    AR_SA = "ar_SA"  # Arabic (Saudi Arabia)
    HI_IN = "hi_IN"  # Hindi (India)
    NL_NL = "nl_NL"  # Dutch (Netherlands)
    SV_SE = "sv_SE"  # Swedish (Sweden)
    NO_NO = "no_NO"  # Norwegian (Norway)
    DA_DK = "da_DK"  # Danish (Denmark)
    FI_FI = "fi_FI"  # Finnish (Finland)


class ComplianceRegion(Enum):
    """Data protection and compliance regions."""
    EU = "eu"        # European Union (GDPR)
    US = "us"        # United States (CCPA, COPPA)
    CA = "ca"        # Canada (PIPEDA)
    AU = "au"        # Australia (Privacy Act)
    UK = "uk"        # United Kingdom (UK GDPR)
    SG = "sg"        # Singapore (PDPA)
    JP = "jp"        # Japan (APPI)
    KR = "kr"        # South Korea (PIPA)
    BR = "br"        # Brazil (LGPD)
    IN = "in"        # India (DPDP Act)
    GLOBAL = "global" # Global compliance baseline


class Platform(Enum):
    """Supported platforms for deployment."""
    LINUX_X64 = "linux_x64"
    LINUX_ARM64 = "linux_arm64"
    WINDOWS_X64 = "windows_x64"
    MACOS_X64 = "macos_x64"
    MACOS_ARM64 = "macos_arm64"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    EDGE = "edge"


@dataclass
class LocalizationEntry:
    """Localization entry for a specific key and locale."""
    key: str
    locale: Locale
    text: str
    context: Optional[str] = None
    plural_forms: Optional[Dict[str, str]] = None
    last_updated: float = field(default_factory=time.time)


@dataclass
class CompliancePolicy:
    """Data protection compliance policy configuration."""
    region: ComplianceRegion
    data_retention_days: int
    consent_required: bool = True
    right_to_delete: bool = True
    right_to_portability: bool = True
    data_minimization: bool = True
    pseudonymization_required: bool = False
    encryption_required: bool = True
    audit_logging: bool = True
    cross_border_transfer_allowed: bool = False
    approved_processors: List[str] = field(default_factory=list)
    
    
@dataclass
class PlatformConfig:
    """Platform-specific configuration."""
    platform: Platform
    cpu_arch: str
    memory_limits: Dict[str, int]
    storage_limits: Dict[str, int]
    network_config: Dict[str, Any]
    security_features: List[str]
    optimization_flags: List[str] = field(default_factory=list)


class LocalizationManager:
    """Manages internationalization and localization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Current locale settings
        self.current_locale = Locale.EN_US
        self.fallback_locale = Locale.EN_US
        
        # Localization data
        self.translations = {}
        self.locale_lock = threading.RLock()
        
        # Initialize from config
        default_locale = self.config.get("i18n", {}).get("default_locale", "en_US")
        self.set_locale(Locale(default_locale))
        
        # Load translations
        self._load_translations()
        
        self.logger.info(f"Localization manager initialized with locale: {self.current_locale.value}")
    
    def _load_translations(self):
        """Load translation files for all supported locales."""
        translations_dir = Path(__file__).parent.parent / "translations"
        
        if not translations_dir.exists():
            self.logger.warning(f"Translations directory not found: {translations_dir}")
            return
        
        for locale in Locale:
            translation_file = translations_dir / f"{locale.value}.json"
            
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self.translations[locale] = json.load(f)
                    
                    self.logger.debug(f"Loaded translations for {locale.value}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load translations for {locale.value}: {e}")
            else:
                # Initialize empty translations
                self.translations[locale] = {}
    
    def set_locale(self, locale: Union[Locale, str]):
        """Set current locale for translations."""
        if isinstance(locale, str):
            try:
                locale = Locale(locale)
            except ValueError:
                self.logger.warning(f"Unsupported locale: {locale}, using fallback")
                locale = self.fallback_locale
        
        with self.locale_lock:
            self.current_locale = locale
            
            # Set system locale if possible
            try:
                locale.setlocale(locale.LC_ALL, locale.value)
            except locale.Error:
                self.logger.debug(f"Could not set system locale to {locale.value}")
        
        self.logger.info(f"Locale set to: {locale.value}")
    
    def get_text(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """
        Get localized text for the given key.
        
        Args:
            key: Translation key
            default: Default text if translation not found
            **kwargs: Variables for text interpolation
            
        Returns:
            Localized text
        """
        with self.locale_lock:
            # Try current locale
            locale_translations = self.translations.get(self.current_locale, {})
            text = locale_translations.get(key)
            
            # Fallback to default locale
            if text is None and self.current_locale != self.fallback_locale:
                fallback_translations = self.translations.get(self.fallback_locale, {})
                text = fallback_translations.get(key)
            
            # Use default or key if no translation found
            if text is None:
                text = default or key
                self.logger.debug(f"Missing translation for key '{key}' in locale '{self.current_locale.value}'")
            
            # Interpolate variables
            if kwargs:
                try:
                    text = text.format(**kwargs)
                except Exception as e:
                    self.logger.warning(f"Failed to interpolate variables in text '{text}': {e}")
            
            return text
    
    def get_plural_text(self, key: str, count: int, default_singular: Optional[str] = None, 
                       default_plural: Optional[str] = None, **kwargs) -> str:
        """
        Get pluralized localized text.
        
        Args:
            key: Translation key
            count: Count for pluralization
            default_singular: Default singular form
            default_plural: Default plural form
            **kwargs: Variables for text interpolation
            
        Returns:
            Pluralized localized text
        """
        # Determine plural form based on locale rules
        if self.current_locale in [Locale.EN_US, Locale.EN_GB]:
            plural_key = f"{key}_plural" if count != 1 else key
        elif self.current_locale in [Locale.RU_RU]:
            # Russian has complex plural rules
            if count % 10 == 1 and count % 100 != 11:
                plural_key = key
            elif 2 <= count % 10 <= 4 and not (12 <= count % 100 <= 14):
                plural_key = f"{key}_few"
            else:
                plural_key = f"{key}_many"
        else:
            # Default pluralization
            plural_key = f"{key}_plural" if count != 1 else key
        
        # Get text with pluralization
        text = self.get_text(plural_key, **kwargs)
        
        # If plural form not found, try singular with count
        if text == plural_key and count != 1:
            text = self.get_text(key, **kwargs)
        
        # Add count to interpolation
        kwargs['count'] = count
        
        try:
            return text.format(**kwargs)
        except:
            return text
    
    def format_number(self, number: Union[int, float]) -> str:
        """Format number according to current locale."""
        try:
            if self.current_locale == Locale.EN_US:
                return f"{number:,}"
            elif self.current_locale in [Locale.FR_FR, Locale.FR_CA]:
                return f"{number:,.2f}".replace(",", " ").replace(".", ",")
            elif self.current_locale in [Locale.DE_DE, Locale.DE_AT]:
                return f"{number:,.2f}".replace(",", ".").replace(".", ",", 1)
            else:
                return f"{number:,}"
        except:
            return str(number)
    
    def format_currency(self, amount: float, currency: str = "USD") -> str:
        """Format currency according to current locale."""
        formatted_amount = self.format_number(amount)
        
        currency_symbols = {
            "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
            "CAD": "C$", "AUD": "A$", "CHF": "CHF", "CNY": "¥"
        }
        
        symbol = currency_symbols.get(currency, currency)
        
        if self.current_locale == Locale.EN_US:
            return f"{symbol}{formatted_amount}"
        elif self.current_locale in [Locale.FR_FR, Locale.FR_CA]:
            return f"{formatted_amount} {symbol}"
        elif self.current_locale in [Locale.DE_DE, Locale.DE_AT]:
            return f"{formatted_amount} {symbol}"
        else:
            return f"{symbol}{formatted_amount}"
    
    def format_date(self, timestamp: float) -> str:
        """Format date according to current locale."""
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        
        if self.current_locale == Locale.EN_US:
            return dt.strftime("%m/%d/%Y")
        elif self.current_locale in [Locale.EN_GB, Locale.FR_FR, Locale.DE_DE]:
            return dt.strftime("%d/%m/%Y")
        elif self.current_locale in [Locale.ZH_CN, Locale.JA_JP, Locale.KO_KR]:
            return dt.strftime("%Y/%m/%d")
        else:
            return dt.strftime("%Y-%m-%d")
    
    def get_supported_locales(self) -> List[str]:
        """Get list of supported locale codes."""
        return [locale.value for locale in Locale]
    
    def add_translation(self, locale: Locale, key: str, text: str):
        """Add or update a translation."""
        with self.locale_lock:
            if locale not in self.translations:
                self.translations[locale] = {}
            
            self.translations[locale][key] = text
        
        self.logger.debug(f"Added translation for {locale.value}: {key}")


class ComplianceManager:
    """Manages data protection and regulatory compliance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Current compliance region
        default_region = self.config.get("compliance", {}).get("region", "global")
        self.current_region = ComplianceRegion(default_region)
        
        # Compliance policies by region
        self.compliance_policies = self._initialize_compliance_policies()
        
        # User consent tracking
        self.user_consents = {}
        self.consent_lock = threading.RLock()
        
        # Data processing audit log
        self.audit_log = []
        
        self.logger.info(f"Compliance manager initialized for region: {self.current_region.value}")
    
    def _initialize_compliance_policies(self) -> Dict[ComplianceRegion, CompliancePolicy]:
        """Initialize compliance policies for different regions."""
        return {
            ComplianceRegion.EU: CompliancePolicy(
                region=ComplianceRegion.EU,
                data_retention_days=365,
                consent_required=True,
                right_to_delete=True,
                right_to_portability=True,
                data_minimization=True,
                pseudonymization_required=True,
                encryption_required=True,
                audit_logging=True,
                cross_border_transfer_allowed=False
            ),
            
            ComplianceRegion.US: CompliancePolicy(
                region=ComplianceRegion.US,
                data_retention_days=1095,  # 3 years
                consent_required=True,
                right_to_delete=True,
                right_to_portability=False,
                data_minimization=False,
                pseudonymization_required=False,
                encryption_required=True,
                audit_logging=True,
                cross_border_transfer_allowed=True
            ),
            
            ComplianceRegion.CA: CompliancePolicy(
                region=ComplianceRegion.CA,
                data_retention_days=730,  # 2 years
                consent_required=True,
                right_to_delete=True,
                right_to_portability=True,
                data_minimization=True,
                pseudonymization_required=False,
                encryption_required=True,
                audit_logging=True,
                cross_border_transfer_allowed=False
            ),
            
            ComplianceRegion.SG: CompliancePolicy(
                region=ComplianceRegion.SG,
                data_retention_days=365,
                consent_required=True,
                right_to_delete=False,
                right_to_portability=False,
                data_minimization=True,
                pseudonymization_required=False,
                encryption_required=True,
                audit_logging=True,
                cross_border_transfer_allowed=True
            ),
            
            ComplianceRegion.GLOBAL: CompliancePolicy(
                region=ComplianceRegion.GLOBAL,
                data_retention_days=365,
                consent_required=True,
                right_to_delete=True,
                right_to_portability=True,
                data_minimization=True,
                pseudonymization_required=True,
                encryption_required=True,
                audit_logging=True,
                cross_border_transfer_allowed=False
            )
        }
    
    def get_compliance_policy(self, region: Optional[ComplianceRegion] = None) -> CompliancePolicy:
        """Get compliance policy for specified region or current region."""
        region = region or self.current_region
        return self.compliance_policies.get(region, self.compliance_policies[ComplianceRegion.GLOBAL])
    
    def check_data_processing_consent(self, user_id: str, processing_type: str) -> bool:
        """
        Check if user has given consent for specific data processing.
        
        Args:
            user_id: User identifier
            processing_type: Type of data processing
            
        Returns:
            True if consent given or not required
        """
        policy = self.get_compliance_policy()
        
        if not policy.consent_required:
            return True
        
        with self.consent_lock:
            user_consent = self.user_consents.get(user_id, {})
            return user_consent.get(processing_type, False)
    
    def record_consent(self, user_id: str, processing_type: str, consent_given: bool, 
                      consent_timestamp: Optional[float] = None):
        """
        Record user consent for data processing.
        
        Args:
            user_id: User identifier
            processing_type: Type of data processing
            consent_given: Whether consent was given
            consent_timestamp: When consent was given
        """
        timestamp = consent_timestamp or time.time()
        
        with self.consent_lock:
            if user_id not in self.user_consents:
                self.user_consents[user_id] = {}
            
            self.user_consents[user_id][processing_type] = consent_given
            self.user_consents[user_id][f"{processing_type}_timestamp"] = timestamp
        
        # Log consent for audit
        self._audit_log_entry(
            action="consent_recorded",
            user_id=user_id,
            details={"processing_type": processing_type, "consent": consent_given}
        )
        
        self.logger.info(f"Recorded consent for user {user_id}: {processing_type} = {consent_given}")
    
    def validate_data_retention(self, data_timestamp: float, data_type: str = "general") -> bool:
        """
        Validate if data can still be retained according to policy.
        
        Args:
            data_timestamp: When data was created
            data_type: Type of data
            
        Returns:
            True if data can be retained
        """
        policy = self.get_compliance_policy()
        current_time = time.time()
        data_age_days = (current_time - data_timestamp) / (24 * 3600)
        
        return data_age_days <= policy.data_retention_days
    
    def process_deletion_request(self, user_id: str) -> Dict[str, Any]:
        """
        Process user's right to deletion (right to be forgotten).
        
        Args:
            user_id: User identifier
            
        Returns:
            Deletion status report
        """
        policy = self.get_compliance_policy()
        
        if not policy.right_to_delete:
            raise ValidationError(f"Right to deletion not supported in region {policy.region.value}")
        
        # Log deletion request
        self._audit_log_entry(
            action="deletion_request",
            user_id=user_id,
            details={"policy_region": policy.region.value}
        )
        
        # In production, this would trigger actual data deletion
        deletion_status = {
            "user_id": user_id,
            "request_timestamp": time.time(),
            "policy_region": policy.region.value,
            "deletion_scheduled": True,
            "estimated_completion": time.time() + (7 * 24 * 3600),  # 7 days
            "affected_systems": ["preferences", "training_data", "logs"]
        }
        
        self.logger.info(f"Deletion request processed for user {user_id}")
        return deletion_status
    
    def generate_data_export(self, user_id: str) -> Dict[str, Any]:
        """
        Generate data export for user (data portability).
        
        Args:
            user_id: User identifier
            
        Returns:
            Exportable user data
        """
        policy = self.get_compliance_policy()
        
        if not policy.right_to_portability:
            raise ValidationError(f"Data portability not supported in region {policy.region.value}")
        
        # Log export request
        self._audit_log_entry(
            action="data_export_request",
            user_id=user_id,
            details={"policy_region": policy.region.value}
        )
        
        # In production, this would collect actual user data
        export_data = {
            "user_id": user_id,
            "export_timestamp": time.time(),
            "policy_region": policy.region.value,
            "data_categories": {
                "preferences": {"preference_pairs": [], "feedback_history": []},
                "training_contributions": {"rlhf_sessions": [], "annotations": []},
                "system_interactions": {"session_logs": [], "performance_metrics": []}
            },
            "export_format": "json",
            "data_retention_days": policy.data_retention_days
        }
        
        self.logger.info(f"Data export generated for user {user_id}")
        return export_data
    
    def _audit_log_entry(self, action: str, user_id: Optional[str] = None, details: Optional[Dict] = None):
        """Add entry to compliance audit log."""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "user_id": user_id,
            "details": details or {},
            "policy_region": self.current_region.value
        }
        
        self.audit_log.append(entry)
        
        # Keep audit log size manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]  # Keep last 5000 entries
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance status report."""
        policy = self.get_compliance_policy()
        
        return {
            "current_region": self.current_region.value,
            "policy_summary": {
                "data_retention_days": policy.data_retention_days,
                "consent_required": policy.consent_required,
                "right_to_delete": policy.right_to_delete,
                "right_to_portability": policy.right_to_portability,
                "encryption_required": policy.encryption_required
            },
            "consent_statistics": {
                "total_users": len(self.user_consents),
                "consent_types_tracked": len(set(
                    key for consents in self.user_consents.values() 
                    for key in consents.keys() if not key.endswith('_timestamp')
                ))
            },
            "audit_log_entries": len(self.audit_log),
            "last_compliance_check": time.time()
        }


class PlatformManager:
    """Manages cross-platform compatibility and optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Detect current platform
        self.current_platform = self._detect_platform()
        
        # Platform-specific configurations
        self.platform_configs = self._initialize_platform_configs()
        
        # Optimization settings
        self.optimizations = {}
        
        self.logger.info(f"Platform manager initialized for: {self.current_platform.value}")
    
    def _detect_platform(self) -> Platform:
        """Detect current platform and architecture."""
        import platform as py_platform
        
        system = py_platform.system().lower()
        machine = py_platform.machine().lower()
        
        # Check if running in containers
        if os.path.exists('/.dockerenv'):
            return Platform.DOCKER
        
        if os.environ.get('KUBERNETES_SERVICE_HOST'):
            return Platform.KUBERNETES
        
        # Cloud platform detection
        if os.environ.get('AWS_REGION'):
            return Platform.AWS
        elif os.environ.get('GOOGLE_CLOUD_PROJECT'):
            return Platform.GCP
        elif os.environ.get('AZURE_CLIENT_ID'):
            return Platform.AZURE
        
        # Local platform detection
        if system == 'linux':
            if 'arm' in machine or 'aarch64' in machine:
                return Platform.LINUX_ARM64
            else:
                return Platform.LINUX_X64
        elif system == 'darwin':
            if 'arm' in machine:
                return Platform.MACOS_ARM64
            else:
                return Platform.MACOS_X64
        elif system == 'windows':
            return Platform.WINDOWS_X64
        
        return Platform.LINUX_X64  # Default fallback
    
    def _initialize_platform_configs(self) -> Dict[Platform, PlatformConfig]:
        """Initialize platform-specific configurations."""
        return {
            Platform.LINUX_X64: PlatformConfig(
                platform=Platform.LINUX_X64,
                cpu_arch="x86_64",
                memory_limits={"quantum_engine": 8192, "cache": 2048},
                storage_limits={"logs": 10240, "cache": 5120},
                network_config={"max_connections": 1000, "timeout": 30},
                security_features=["seccomp", "apparmor", "selinux"],
                optimization_flags=["avx2", "sse4", "native"]
            ),
            
            Platform.LINUX_ARM64: PlatformConfig(
                platform=Platform.LINUX_ARM64,
                cpu_arch="aarch64",
                memory_limits={"quantum_engine": 4096, "cache": 1024},
                storage_limits={"logs": 5120, "cache": 2048},
                network_config={"max_connections": 500, "timeout": 30},
                security_features=["seccomp", "apparmor"],
                optimization_flags=["neon", "native"]
            ),
            
            Platform.DOCKER: PlatformConfig(
                platform=Platform.DOCKER,
                cpu_arch="auto",
                memory_limits={"quantum_engine": 6144, "cache": 1536},
                storage_limits={"logs": 2048, "cache": 1024},
                network_config={"max_connections": 500, "timeout": 45},
                security_features=["user_namespaces", "cgroups"],
                optimization_flags=["container_optimized"]
            ),
            
            Platform.KUBERNETES: PlatformConfig(
                platform=Platform.KUBERNETES,
                cpu_arch="auto",
                memory_limits={"quantum_engine": 8192, "cache": 2048},
                storage_limits={"logs": 5120, "cache": 2048},
                network_config={"max_connections": 1000, "timeout": 60},
                security_features=["pod_security", "network_policies", "rbac"],
                optimization_flags=["k8s_optimized", "horizontal_scaling"]
            ),
            
            Platform.AWS: PlatformConfig(
                platform=Platform.AWS,
                cpu_arch="auto",
                memory_limits={"quantum_engine": 16384, "cache": 4096},
                storage_limits={"logs": 20480, "cache": 10240},
                network_config={"max_connections": 2000, "timeout": 30},
                security_features=["iam", "vpc", "encryption", "cloudtrail"],
                optimization_flags=["aws_graviton", "nitro_optimized"]
            )
        }
    
    def get_platform_config(self, platform: Optional[Platform] = None) -> PlatformConfig:
        """Get configuration for specified platform or current platform."""
        platform = platform or self.current_platform
        
        config = self.platform_configs.get(platform)
        if not config:
            # Return generic Linux config as fallback
            config = self.platform_configs[Platform.LINUX_X64]
            self.logger.warning(f"No config found for {platform.value}, using Linux x64 config")
        
        return config
    
    def apply_platform_optimizations(self):
        """Apply platform-specific optimizations."""
        config = self.get_platform_config()
        
        # Apply memory optimizations
        os.environ['QUANTUM_ENGINE_MEMORY_LIMIT'] = str(config.memory_limits["quantum_engine"])
        
        # Apply CPU optimizations
        if "avx2" in config.optimization_flags:
            os.environ['USE_AVX2'] = "1"
        
        if "native" in config.optimization_flags:
            os.environ['NATIVE_OPTIMIZATION'] = "1"
        
        # Apply network optimizations
        max_conn = config.network_config["max_connections"]
        os.environ['MAX_NETWORK_CONNECTIONS'] = str(max_conn)
        
        self.logger.info(f"Applied optimizations for {config.platform.value}")
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get current system capabilities and constraints."""
        import psutil
        
        config = self.get_platform_config()
        
        return {
            "platform": self.current_platform.value,
            "cpu_arch": config.cpu_arch,
            "cpu_cores": os.cpu_count(),
            "memory_total_mb": psutil.virtual_memory().total // (1024 * 1024),
            "memory_available_mb": psutil.virtual_memory().available // (1024 * 1024),
            "disk_space_gb": psutil.disk_usage('/').free // (1024 * 1024 * 1024),
            "security_features": config.security_features,
            "optimization_flags": config.optimization_flags,
            "network_config": config.network_config
        }


class GlobalizationManager:
    """Master class coordinating all global-first features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Initialize sub-managers
        self.localization = LocalizationManager(config)
        self.compliance = ComplianceManager(config)
        self.platform = PlatformManager(config)
        
        # Global settings
        self.timezone = self.config.get("global", {}).get("timezone", "UTC")
        
        # Apply platform optimizations
        self.platform.apply_platform_optimizations()
        
        self.logger.info("Globalization manager initialized - ready for worldwide deployment")
    
    def get_localized_text(self, key: str, **kwargs) -> str:
        """Get localized text with current locale."""
        return self.localization.get_text(key, **kwargs)
    
    def check_compliance(self, operation: str, user_id: Optional[str] = None) -> bool:
        """Check if operation complies with current region's policies."""
        if user_id:
            return self.compliance.check_data_processing_consent(user_id, operation)
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for global deployment."""
        return {
            "localization": {
                "current_locale": self.localization.current_locale.value,
                "supported_locales": len(self.localization.get_supported_locales())
            },
            "compliance": {
                "region": self.compliance.current_region.value,
                "policies_active": len(self.compliance.compliance_policies)
            },
            "platform": {
                "current_platform": self.platform.current_platform.value,
                "capabilities": self.platform.get_system_capabilities()
            },
            "global_ready": True,
            "deployment_regions": ["us", "eu", "ca", "sg", "au", "jp", "br"]
        }
    
    def validate_global_deployment_readiness(self) -> Dict[str, Any]:
        """Validate system readiness for global deployment."""
        checks = {
            "i18n_ready": len(self.localization.translations) > 1,
            "compliance_configured": len(self.compliance.compliance_policies) >= 5,
            "platform_optimized": bool(self.platform.current_platform),
            "security_features_enabled": len(self.platform.get_platform_config().security_features) > 0,
            "multi_region_support": True
        }
        
        all_ready = all(checks.values())
        
        return {
            "ready_for_global_deployment": all_ready,
            "readiness_checks": checks,
            "supported_regions": [region.value for region in ComplianceRegion],
            "supported_locales": self.localization.get_supported_locales(),
            "deployment_platforms": [platform.value for platform in Platform]
        }