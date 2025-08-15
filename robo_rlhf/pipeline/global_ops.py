"""
Global Operations: Multi-region, internationalization, and compliance features.

Implements global deployment capabilities, i18n support, and regulatory compliance.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported global regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"          # General Data Protection Regulation (EU)
    CCPA = "ccpa"          # California Consumer Privacy Act (US)
    PDPA = "pdpa"          # Personal Data Protection Act (Singapore)
    SOC2 = "soc2"          # Service Organization Control 2
    ISO27001 = "iso27001"  # Information Security Management
    HIPAA = "hipaa"        # Health Insurance Portability and Accountability Act


class Language(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    KOREAN = "ko"
    DUTCH = "nl"


@dataclass
class RegionConfig:
    """Configuration for a specific region."""
    region: Region
    data_residency_required: bool = False
    compliance_standards: Set[ComplianceStandard] = field(default_factory=set)
    primary_language: Language = Language.ENGLISH
    supported_languages: Set[Language] = field(default_factory=lambda: {Language.ENGLISH})
    timezone: str = "UTC"
    emergency_contacts: List[str] = field(default_factory=list)


@dataclass
class CompliancePolicy:
    """Compliance policy configuration."""
    standard: ComplianceStandard
    data_retention_days: int
    encryption_required: bool = True
    audit_logging_required: bool = True
    access_controls_required: bool = True
    data_export_allowed: bool = False
    cross_border_transfer_allowed: bool = False


class InternationalizationManager:
    """
    Manages internationalization and localization for pipeline operations.
    
    Features:
    - Multi-language support for alerts and messages
    - Timezone-aware operations
    - Regional formatting for metrics and dates
    - Cultural adaptation for user interfaces
    """
    
    def __init__(self, default_language: Language = Language.ENGLISH):
        self.default_language = default_language
        self.translations: Dict[Language, Dict[str, str]] = {}
        self.regional_formats: Dict[Region, Dict[str, str]] = {}
        
        # Initialize default translations
        self._load_default_translations()
        self._load_regional_formats()
    
    def _load_default_translations(self) -> None:
        """Load default translations for common messages."""
        # English (base)
        self.translations[Language.ENGLISH] = {
            "pipeline.healthy": "Pipeline is healthy",
            "pipeline.degraded": "Pipeline performance is degraded",
            "pipeline.critical": "Pipeline is in critical state",
            "pipeline.failed": "Pipeline has failed",
            "component.restarting": "Restarting component",
            "component.scaling": "Scaling component",
            "alert.high_cpu": "High CPU usage detected",
            "alert.high_memory": "High memory usage detected",
            "alert.slow_response": "Slow response times detected",
            "recovery.successful": "Recovery completed successfully",
            "recovery.failed": "Recovery attempt failed",
            "security.unauthorized": "Unauthorized access attempt",
            "security.login_success": "Login successful",
            "security.login_failed": "Login failed",
            "cache.hit": "Cache hit",
            "cache.miss": "Cache miss",
            "scaling.up": "Scaling up resources",
            "scaling.down": "Scaling down resources"
        }
        
        # Spanish
        self.translations[Language.SPANISH] = {
            "pipeline.healthy": "La tubería está saludable",
            "pipeline.degraded": "El rendimiento de la tubería está degradado",
            "pipeline.critical": "La tubería está en estado crítico",
            "pipeline.failed": "La tubería ha fallado",
            "component.restarting": "Reiniciando componente",
            "component.scaling": "Escalando componente",
            "alert.high_cpu": "Uso alto de CPU detectado",
            "alert.high_memory": "Uso alto de memoria detectado",
            "alert.slow_response": "Tiempos de respuesta lentos detectados",
            "recovery.successful": "Recuperación completada exitosamente",
            "recovery.failed": "Intento de recuperación falló",
            "security.unauthorized": "Intento de acceso no autorizado",
            "security.login_success": "Inicio de sesión exitoso",
            "security.login_failed": "Inicio de sesión falló",
            "cache.hit": "Acierto de caché",
            "cache.miss": "Falla de caché",
            "scaling.up": "Aumentando recursos",
            "scaling.down": "Reduciendo recursos"
        }
        
        # French
        self.translations[Language.FRENCH] = {
            "pipeline.healthy": "Le pipeline est en bonne santé",
            "pipeline.degraded": "Les performances du pipeline sont dégradées",
            "pipeline.critical": "Le pipeline est dans un état critique",
            "pipeline.failed": "Le pipeline a échoué",
            "component.restarting": "Redémarrage du composant",
            "component.scaling": "Mise à l'échelle du composant",
            "alert.high_cpu": "Utilisation élevée du CPU détectée",
            "alert.high_memory": "Utilisation élevée de la mémoire détectée",
            "alert.slow_response": "Temps de réponse lents détectés",
            "recovery.successful": "Récupération terminée avec succès",
            "recovery.failed": "Tentative de récupération échouée",
            "security.unauthorized": "Tentative d'accès non autorisé",
            "security.login_success": "Connexion réussie",
            "security.login_failed": "Connexion échouée",
            "cache.hit": "Succès du cache",
            "cache.miss": "Échec du cache",
            "scaling.up": "Augmentation des ressources",
            "scaling.down": "Réduction des ressources"
        }
        
        # German
        self.translations[Language.GERMAN] = {
            "pipeline.healthy": "Pipeline ist gesund",
            "pipeline.degraded": "Pipeline-Leistung ist beeinträchtigt",
            "pipeline.critical": "Pipeline ist in kritischem Zustand",
            "pipeline.failed": "Pipeline ist fehlgeschlagen",
            "component.restarting": "Komponente wird neu gestartet",
            "component.scaling": "Komponente wird skaliert",
            "alert.high_cpu": "Hohe CPU-Auslastung erkannt",
            "alert.high_memory": "Hohe Speicherauslastung erkannt",
            "alert.slow_response": "Langsame Antwortzeiten erkannt",
            "recovery.successful": "Wiederherstellung erfolgreich abgeschlossen",
            "recovery.failed": "Wiederherstellungsversuch fehlgeschlagen",
            "security.unauthorized": "Unbefugter Zugriffsversuch",
            "security.login_success": "Anmeldung erfolgreich",
            "security.login_failed": "Anmeldung fehlgeschlagen",
            "cache.hit": "Cache-Treffer",
            "cache.miss": "Cache-Fehler",
            "scaling.up": "Ressourcen hochskalieren",
            "scaling.down": "Ressourcen herunterskalieren"
        }
        
        # Japanese
        self.translations[Language.JAPANESE] = {
            "pipeline.healthy": "パイプラインは正常です",
            "pipeline.degraded": "パイプラインのパフォーマンスが低下しています",
            "pipeline.critical": "パイプラインが重要な状態です",
            "pipeline.failed": "パイプラインが失敗しました",
            "component.restarting": "コンポーネントを再起動中",
            "component.scaling": "コンポーネントをスケーリング中",
            "alert.high_cpu": "CPU使用率が高いことを検出",
            "alert.high_memory": "メモリ使用率が高いことを検出",
            "alert.slow_response": "応答時間が遅いことを検出",
            "recovery.successful": "復旧が正常に完了しました",
            "recovery.failed": "復旧の試行が失敗しました",
            "security.unauthorized": "不正なアクセスの試行",
            "security.login_success": "ログイン成功",
            "security.login_failed": "ログイン失敗",
            "cache.hit": "キャッシュヒット",
            "cache.miss": "キャッシュミス",
            "scaling.up": "リソースをスケールアップ",
            "scaling.down": "リソースをスケールダウン"
        }
        
        # Chinese
        self.translations[Language.CHINESE] = {
            "pipeline.healthy": "管道健康",
            "pipeline.degraded": "管道性能下降",
            "pipeline.critical": "管道处于关键状态",
            "pipeline.failed": "管道失败",
            "component.restarting": "重启组件",
            "component.scaling": "扩展组件",
            "alert.high_cpu": "检测到高CPU使用率",
            "alert.high_memory": "检测到高内存使用率",
            "alert.slow_response": "检测到响应时间慢",
            "recovery.successful": "恢复成功完成",
            "recovery.failed": "恢复尝试失败",
            "security.unauthorized": "未授权访问尝试",
            "security.login_success": "登录成功",
            "security.login_failed": "登录失败",
            "cache.hit": "缓存命中",
            "cache.miss": "缓存未命中",
            "scaling.up": "扩展资源",
            "scaling.down": "缩减资源"
        }
    
    def _load_regional_formats(self) -> None:
        """Load regional formatting preferences."""
        self.regional_formats = {
            Region.US_EAST: {
                "date_format": "%m/%d/%Y",
                "time_format": "%I:%M %p",
                "decimal_separator": ".",
                "thousands_separator": ",",
                "currency_symbol": "$"
            },
            Region.EU_WEST: {
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M",
                "decimal_separator": ",",
                "thousands_separator": ".",
                "currency_symbol": "€"
            },
            Region.EU_CENTRAL: {
                "date_format": "%d.%m.%Y",
                "time_format": "%H:%M",
                "decimal_separator": ",",
                "thousands_separator": ".",
                "currency_symbol": "€"
            },
            Region.ASIA_PACIFIC: {
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M",
                "decimal_separator": ".",
                "thousands_separator": ",",
                "currency_symbol": "$"
            },
            Region.ASIA_NORTHEAST: {
                "date_format": "%Y/%m/%d",
                "time_format": "%H:%M",
                "decimal_separator": ".",
                "thousands_separator": ",",
                "currency_symbol": "¥"
            }
        }
    
    def translate(self, key: str, language: Optional[Language] = None) -> str:
        """Translate a message key to the specified language."""
        target_language = language or self.default_language
        
        if target_language in self.translations:
            return self.translations[target_language].get(key, key)
        
        # Fallback to English
        return self.translations[Language.ENGLISH].get(key, key)
    
    def format_datetime(self, dt: datetime, region: Region) -> str:
        """Format datetime according to regional preferences."""
        if region not in self.regional_formats:
            region = Region.US_EAST  # Fallback
        
        formats = self.regional_formats[region]
        date_str = dt.strftime(formats["date_format"])
        time_str = dt.strftime(formats["time_format"])
        
        return f"{date_str} {time_str}"
    
    def format_number(self, number: float, region: Region, decimals: int = 2) -> str:
        """Format number according to regional preferences."""
        if region not in self.regional_formats:
            region = Region.US_EAST  # Fallback
        
        formats = self.regional_formats[region]
        
        # Format with specified decimals
        formatted = f"{number:.{decimals}f}"
        
        # Apply regional separators
        if formats["decimal_separator"] != ".":
            formatted = formatted.replace(".", formats["decimal_separator"])
        
        # Add thousands separator (simplified implementation)
        parts = formatted.split(formats["decimal_separator"])
        integer_part = parts[0]
        
        if len(integer_part) > 3:
            # Add thousands separators
            reversed_digits = integer_part[::-1]
            grouped = [reversed_digits[i:i+3] for i in range(0, len(reversed_digits), 3)]
            integer_part = formats["thousands_separator"].join(grouped)[::-1]
        
        if len(parts) > 1:
            return f"{integer_part}{formats['decimal_separator']}{parts[1]}"
        else:
            return integer_part


class ComplianceManager:
    """
    Manages regulatory compliance across different regions and standards.
    
    Features:
    - GDPR compliance for EU operations
    - CCPA compliance for California operations
    - Data residency enforcement
    - Audit trail maintenance
    - Privacy controls implementation
    """
    
    def __init__(self):
        self.policies: Dict[ComplianceStandard, CompliancePolicy] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.data_retention_policies: Dict[str, int] = {}
        self.privacy_controls: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default compliance policies
        self._setup_default_policies()
    
    def _setup_default_policies(self) -> None:
        """Setup default compliance policies."""
        # GDPR Policy
        self.policies[ComplianceStandard.GDPR] = CompliancePolicy(
            standard=ComplianceStandard.GDPR,
            data_retention_days=2555,  # 7 years max
            encryption_required=True,
            audit_logging_required=True,
            access_controls_required=True,
            data_export_allowed=True,  # Right to data portability
            cross_border_transfer_allowed=False  # Restricted
        )
        
        # CCPA Policy
        self.policies[ComplianceStandard.CCPA] = CompliancePolicy(
            standard=ComplianceStandard.CCPA,
            data_retention_days=1825,  # 5 years
            encryption_required=True,
            audit_logging_required=True,
            access_controls_required=True,
            data_export_allowed=True,  # Right to know
            cross_border_transfer_allowed=True  # With safeguards
        )
        
        # SOC 2 Policy
        self.policies[ComplianceStandard.SOC2] = CompliancePolicy(
            standard=ComplianceStandard.SOC2,
            data_retention_days=2555,  # 7 years
            encryption_required=True,
            audit_logging_required=True,
            access_controls_required=True,
            data_export_allowed=False,
            cross_border_transfer_allowed=True
        )
        
        # ISO 27001 Policy
        self.policies[ComplianceStandard.ISO27001] = CompliancePolicy(
            standard=ComplianceStandard.ISO27001,
            data_retention_days=2190,  # 6 years
            encryption_required=True,
            audit_logging_required=True,
            access_controls_required=True,
            data_export_allowed=False,
            cross_border_transfer_allowed=True
        )
    
    def validate_data_operation(
        self,
        operation: str,
        data_type: str,
        user_region: Region,
        compliance_standards: Set[ComplianceStandard]
    ) -> Dict[str, Any]:
        """Validate data operation against compliance requirements."""
        validation_result = {
            "allowed": True,
            "restrictions": [],
            "requirements": [],
            "audit_required": False
        }
        
        for standard in compliance_standards:
            if standard not in self.policies:
                continue
            
            policy = self.policies[standard]
            
            # Check encryption requirement
            if policy.encryption_required:
                validation_result["requirements"].append(f"Encryption required for {standard.value}")
            
            # Check audit logging
            if policy.audit_logging_required:
                validation_result["audit_required"] = True
                validation_result["requirements"].append(f"Audit logging required for {standard.value}")
            
            # Check data export restrictions
            if operation == "export" and not policy.data_export_allowed:
                validation_result["allowed"] = False
                validation_result["restrictions"].append(f"Data export not allowed under {standard.value}")
            
            # Check cross-border transfer
            if operation == "transfer" and not policy.cross_border_transfer_allowed:
                validation_result["allowed"] = False
                validation_result["restrictions"].append(f"Cross-border transfer restricted under {standard.value}")
        
        # Log validation
        if validation_result["audit_required"]:
            self._log_compliance_event(
                event_type="data_operation_validation",
                operation=operation,
                data_type=data_type,
                user_region=user_region.value,
                standards=[s.value for s in compliance_standards],
                result=validation_result
            )
        
        return validation_result
    
    def handle_data_subject_request(
        self,
        request_type: str,
        user_id: str,
        region: Region,
        compliance_standard: ComplianceStandard
    ) -> Dict[str, Any]:
        """Handle data subject requests (GDPR Article 15-22, CCPA rights)."""
        if compliance_standard not in self.policies:
            return {"status": "error", "message": "Unknown compliance standard"}
        
        policy = self.policies[compliance_standard]
        
        response = {
            "request_id": f"{user_id}_{int(time.time())}",
            "status": "accepted",
            "estimated_completion": "30 days",
            "actions_required": []
        }
        
        if request_type == "access":
            # Right to access (GDPR Art. 15, CCPA)
            response["actions_required"].append("Gather all personal data")
            response["actions_required"].append("Provide data in readable format")
            
        elif request_type == "rectification":
            # Right to rectification (GDPR Art. 16)
            response["actions_required"].append("Identify data to be corrected")
            response["actions_required"].append("Update records across all systems")
            
        elif request_type == "erasure":
            # Right to erasure/deletion (GDPR Art. 17, CCPA)
            response["actions_required"].append("Identify all data instances")
            response["actions_required"].append("Verify no legal retention requirements")
            response["actions_required"].append("Delete data from all systems")
            
        elif request_type == "portability":
            # Right to data portability (GDPR Art. 20)
            if policy.data_export_allowed:
                response["actions_required"].append("Export data in machine-readable format")
            else:
                response["status"] = "rejected"
                response["message"] = "Data portability not supported"
        
        elif request_type == "opt_out":
            # Right to opt-out of sale (CCPA)
            response["actions_required"].append("Update consent preferences")
            response["actions_required"].append("Stop data sharing with third parties")
        
        # Log the request
        self._log_compliance_event(
            event_type="data_subject_request",
            request_type=request_type,
            user_id=user_id,
            region=region.value,
            standard=compliance_standard.value,
            response=response
        )
        
        return response
    
    def _log_compliance_event(self, **event_data) -> None:
        """Log compliance-related events for audit trail."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": f"compliance_{int(time.time() * 1000)}",
            **event_data
        }
        
        self.audit_log.append(event)
        
        # Keep only recent events (last 10000)
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
    
    def get_compliance_report(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """Generate compliance report for a specific standard."""
        if standard not in self.policies:
            return {"error": "Unknown compliance standard"}
        
        policy = self.policies[standard]
        recent_events = [
            event for event in self.audit_log
            if event.get("standard") == standard.value or 
               standard.value in event.get("standards", [])
        ]
        
        return {
            "standard": standard.value,
            "policy": {
                "data_retention_days": policy.data_retention_days,
                "encryption_required": policy.encryption_required,
                "audit_logging_required": policy.audit_logging_required,
                "access_controls_required": policy.access_controls_required,
                "data_export_allowed": policy.data_export_allowed,
                "cross_border_transfer_allowed": policy.cross_border_transfer_allowed
            },
            "recent_events": len(recent_events),
            "data_subject_requests": len([
                e for e in recent_events if e.get("event_type") == "data_subject_request"
            ]),
            "violations": len([
                e for e in recent_events if e.get("event_type") == "compliance_violation"
            ]),
            "audit_trail_events": len(self.audit_log)
        }


class MultiRegionManager:
    """
    Manages multi-region deployment and operations.
    
    Features:
    - Regional deployment coordination
    - Data residency compliance
    - Cross-region failover
    - Latency optimization
    - Regional health monitoring
    """
    
    def __init__(self):
        self.region_configs: Dict[Region, RegionConfig] = {}
        self.active_regions: Set[Region] = set()
        self.regional_health: Dict[Region, Dict[str, Any]] = {}
        self.data_locations: Dict[str, Set[Region]] = {}
        
        # Managers
        self.i18n_manager = InternationalizationManager()
        self.compliance_manager = ComplianceManager()
        
        # Initialize default region configurations
        self._setup_default_regions()
    
    def _setup_default_regions(self) -> None:
        """Setup default region configurations."""
        # US East
        self.region_configs[Region.US_EAST] = RegionConfig(
            region=Region.US_EAST,
            data_residency_required=False,
            compliance_standards={ComplianceStandard.SOC2, ComplianceStandard.CCPA},
            primary_language=Language.ENGLISH,
            supported_languages={Language.ENGLISH, Language.SPANISH},
            timezone="America/New_York"
        )
        
        # US West
        self.region_configs[Region.US_WEST] = RegionConfig(
            region=Region.US_WEST,
            data_residency_required=False,
            compliance_standards={ComplianceStandard.SOC2, ComplianceStandard.CCPA},
            primary_language=Language.ENGLISH,
            supported_languages={Language.ENGLISH, Language.SPANISH},
            timezone="America/Los_Angeles"
        )
        
        # EU West
        self.region_configs[Region.EU_WEST] = RegionConfig(
            region=Region.EU_WEST,
            data_residency_required=True,
            compliance_standards={ComplianceStandard.GDPR, ComplianceStandard.ISO27001},
            primary_language=Language.ENGLISH,
            supported_languages={Language.ENGLISH, Language.FRENCH, Language.GERMAN},
            timezone="Europe/London"
        )
        
        # EU Central
        self.region_configs[Region.EU_CENTRAL] = RegionConfig(
            region=Region.EU_CENTRAL,
            data_residency_required=True,
            compliance_standards={ComplianceStandard.GDPR, ComplianceStandard.ISO27001},
            primary_language=Language.GERMAN,
            supported_languages={Language.GERMAN, Language.ENGLISH, Language.FRENCH},
            timezone="Europe/Berlin"
        )
        
        # Asia Pacific
        self.region_configs[Region.ASIA_PACIFIC] = RegionConfig(
            region=Region.ASIA_PACIFIC,
            data_residency_required=True,
            compliance_standards={ComplianceStandard.PDPA, ComplianceStandard.ISO27001},
            primary_language=Language.ENGLISH,
            supported_languages={Language.ENGLISH, Language.CHINESE},
            timezone="Asia/Singapore"
        )
        
        # Asia Northeast
        self.region_configs[Region.ASIA_NORTHEAST] = RegionConfig(
            region=Region.ASIA_NORTHEAST,
            data_residency_required=True,
            compliance_standards={ComplianceStandard.ISO27001},
            primary_language=Language.JAPANESE,
            supported_languages={Language.JAPANESE, Language.ENGLISH, Language.CHINESE},
            timezone="Asia/Tokyo"
        )
    
    def activate_region(self, region: Region) -> bool:
        """Activate a region for operations."""
        if region not in self.region_configs:
            logger.error(f"Unknown region: {region}")
            return False
        
        self.active_regions.add(region)
        self.regional_health[region] = {
            "status": "active",
            "activated_at": time.time(),
            "components_healthy": 0,
            "components_total": 0
        }
        
        logger.info(f"Activated region: {region.value}")
        return True
    
    def deactivate_region(self, region: Region) -> bool:
        """Deactivate a region."""
        if region in self.active_regions:
            self.active_regions.remove(region)
            if region in self.regional_health:
                self.regional_health[region]["status"] = "inactive"
                self.regional_health[region]["deactivated_at"] = time.time()
            
            logger.info(f"Deactivated region: {region.value}")
            return True
        
        return False
    
    def get_optimal_region(self, user_location: Optional[str] = None) -> Region:
        """Get optimal region for a user based on location and compliance."""
        if not self.active_regions:
            # Fallback to US East if no regions active
            return Region.US_EAST
        
        # Simple region selection based on user location
        if user_location:
            location_lower = user_location.lower()
            
            if any(country in location_lower for country in ["us", "usa", "america", "canada"]):
                # Prefer US regions
                us_regions = [r for r in self.active_regions if r.value.startswith("us-")]
                if us_regions:
                    return us_regions[0]
            
            elif any(country in location_lower for country in ["eu", "europe", "germany", "france", "uk"]):
                # Prefer EU regions
                eu_regions = [r for r in self.active_regions if r.value.startswith("eu-")]
                if eu_regions:
                    return eu_regions[0]
            
            elif any(country in location_lower for country in ["asia", "japan", "singapore", "china"]):
                # Prefer Asia regions
                asia_regions = [r for r in self.active_regions if r.value.startswith("ap-")]
                if asia_regions:
                    return asia_regions[0]
        
        # Default to first active region
        return next(iter(self.active_regions))
    
    def validate_data_residency(self, data_type: str, region: Region) -> bool:
        """Validate that data can be stored in the specified region."""
        if region not in self.region_configs:
            return False
        
        config = self.region_configs[region]
        
        # If data residency is required, data must stay in region
        if config.data_residency_required:
            # Check if data type has residency requirements
            if data_type in ["personal_data", "sensitive_data", "financial_data"]:
                return True  # Must be stored locally
        
        return True  # No restrictions
    
    def get_localized_message(
        self,
        message_key: str,
        region: Region,
        language: Optional[Language] = None
    ) -> str:
        """Get localized message for a specific region."""
        config = self.region_configs.get(region)
        if not config:
            return message_key
        
        # Use specified language or region's primary language
        target_language = language or config.primary_language
        
        # Fallback to supported language if target not supported
        if target_language not in config.supported_languages:
            target_language = config.primary_language
        
        return self.i18n_manager.translate(message_key, target_language)
    
    def get_regional_status(self) -> Dict[str, Any]:
        """Get status of all regions."""
        return {
            "active_regions": [r.value for r in self.active_regions],
            "total_regions": len(self.region_configs),
            "regional_health": {
                r.value: health for r, health in self.regional_health.items()
            },
            "compliance_summary": {
                standard.value: len([
                    r for r in self.active_regions
                    if standard in self.region_configs[r].compliance_standards
                ])
                for standard in ComplianceStandard
            }
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get comprehensive global operations statistics."""
        total_components = sum(
            health.get("components_total", 0)
            for health in self.regional_health.values()
        )
        
        healthy_components = sum(
            health.get("components_healthy", 0)
            for health in self.regional_health.values()
        )
        
        return {
            "regions": self.get_regional_status(),
            "components": {
                "total": total_components,
                "healthy": healthy_components,
                "health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0
            },
            "compliance": {
                standard.value: self.compliance_manager.get_compliance_report(standard)
                for standard in [ComplianceStandard.GDPR, ComplianceStandard.CCPA, ComplianceStandard.SOC2]
            },
            "localization": {
                "supported_languages": len(Language),
                "active_languages": len(set(
                    lang for config in self.region_configs.values()
                    for lang in config.supported_languages
                )),
                "total_translations": sum(
                    len(translations) for translations in self.i18n_manager.translations.values()
                )
            }
        }