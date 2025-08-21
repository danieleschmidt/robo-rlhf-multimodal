#!/usr/bin/env python3
"""
Global-First Implementation
===========================

Comprehensive internationalization, localization, and global compliance features
for worldwide deployment and regulatory adherence.
"""

import asyncio
import logging
import time
import json
import sys
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import locale as system_locale
import gettext
from datetime import datetime, timezone
import re

# Core imports
from robo_rlhf.core.logging import setup_logging, get_logger


class SupportedLocale(Enum):
    """Supported locales for internationalization."""
    EN_US = "en_US"  # English (United States)
    ES_ES = "es_ES"  # Spanish (Spain)
    FR_FR = "fr_FR"  # French (France)
    DE_DE = "de_DE"  # German (Germany)
    JA_JP = "ja_JP"  # Japanese (Japan)
    ZH_CN = "zh_CN"  # Chinese (Simplified)
    PT_BR = "pt_BR"  # Portuguese (Brazil)
    IT_IT = "it_IT"  # Italian (Italy)
    RU_RU = "ru_RU"  # Russian (Russia)
    KO_KR = "ko_KR"  # Korean (South Korea)


class ComplianceRegion(Enum):
    """Compliance regions and their regulations."""
    EU = "eu"           # GDPR, AI Act
    US = "us"           # CCPA, SOX
    APAC = "apac"       # Various APAC regulations
    CANADA = "canada"   # PIPEDA
    BRAZIL = "brazil"   # LGPD
    GLOBAL = "global"   # Global standards


class DataClassification(Enum):
    """Data classification levels for compliance."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"                    # Personally Identifiable Information
    SENSITIVE_PII = "sensitive_pii" # Sensitive PII (SSN, financial, etc.)


@dataclass
class LocalizationConfig:
    """Localization configuration."""
    locale: SupportedLocale
    language_code: str
    country_code: str
    currency: str
    date_format: str
    time_format: str
    number_format: str
    rtl: bool = False  # Right-to-left text direction


@dataclass
class ComplianceRequirement:
    """Compliance requirement definition."""
    region: ComplianceRegion
    regulation_name: str
    requirement_id: str
    description: str
    data_types: List[DataClassification]
    mandatory: bool
    implementation_status: str


class GlobalizationManager:
    """
    Comprehensive globalization manager handling internationalization,
    localization, and compliance for worldwide deployment.
    """
    
    def __init__(self, default_locale: SupportedLocale = SupportedLocale.EN_US):
        """Initialize globalization manager."""
        self.default_locale = default_locale
        self.current_locale = default_locale
        
        # Initialize logging
        setup_logging(level="INFO")
        self.logger = get_logger(__name__)
        
        # Localization configurations
        self.locale_configs = self._initialize_locale_configs()
        self.translations = self._initialize_translations()
        
        # Compliance requirements
        self.compliance_requirements = self._initialize_compliance_requirements()
        self.active_regions = [ComplianceRegion.GLOBAL]
        
        # Regional settings
        self.regional_settings = self._initialize_regional_settings()
        
        self.logger.info(f"🌍 Globalization Manager initialized (locale: {default_locale.value})")
    
    def _initialize_locale_configs(self) -> Dict[SupportedLocale, LocalizationConfig]:
        """Initialize localization configurations for all supported locales."""
        return {
            SupportedLocale.EN_US: LocalizationConfig(
                locale=SupportedLocale.EN_US,
                language_code="en",
                country_code="US",
                currency="USD",
                date_format="%m/%d/%Y",
                time_format="%I:%M %p",
                number_format="1,234.56"
            ),
            SupportedLocale.ES_ES: LocalizationConfig(
                locale=SupportedLocale.ES_ES,
                language_code="es",
                country_code="ES",
                currency="EUR",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1.234,56"
            ),
            SupportedLocale.FR_FR: LocalizationConfig(
                locale=SupportedLocale.FR_FR,
                language_code="fr",
                country_code="FR",
                currency="EUR",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1 234,56"
            ),
            SupportedLocale.DE_DE: LocalizationConfig(
                locale=SupportedLocale.DE_DE,
                language_code="de",
                country_code="DE",
                currency="EUR",
                date_format="%d.%m.%Y",
                time_format="%H:%M",
                number_format="1.234,56"
            ),
            SupportedLocale.JA_JP: LocalizationConfig(
                locale=SupportedLocale.JA_JP,
                language_code="ja",
                country_code="JP",
                currency="JPY",
                date_format="%Y年%m月%d日",
                time_format="%H:%M",
                number_format="1,234"
            ),
            SupportedLocale.ZH_CN: LocalizationConfig(
                locale=SupportedLocale.ZH_CN,
                language_code="zh",
                country_code="CN",
                currency="CNY",
                date_format="%Y年%m月%d日",
                time_format="%H:%M",
                number_format="1,234.56"
            ),
            SupportedLocale.PT_BR: LocalizationConfig(
                locale=SupportedLocale.PT_BR,
                language_code="pt",
                country_code="BR",
                currency="BRL",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1.234,56"
            )
        }
    
    def _initialize_translations(self) -> Dict[SupportedLocale, Dict[str, str]]:
        """Initialize translation dictionaries for all supported locales."""
        translations = {
            SupportedLocale.EN_US: {
                "welcome": "Welcome to Robo-RLHF-Multimodal",
                "training_started": "Training started",
                "training_completed": "Training completed successfully",
                "error_occurred": "An error occurred",
                "data_collection": "Data Collection",
                "preference_learning": "Preference Learning",
                "policy_training": "Policy Training",
                "evaluation": "Evaluation",
                "autonomous_sdlc": "Autonomous SDLC",
                "quality_gates": "Quality Gates",
                "deployment": "Deployment",
                "monitoring": "Monitoring",
                "security_scan": "Security Scan",
                "performance_test": "Performance Test",
                "compliance_check": "Compliance Check"
            },
            SupportedLocale.ES_ES: {
                "welcome": "Bienvenido a Robo-RLHF-Multimodal",
                "training_started": "Entrenamiento iniciado",
                "training_completed": "Entrenamiento completado exitosamente",
                "error_occurred": "Ocurrió un error",
                "data_collection": "Recolección de Datos",
                "preference_learning": "Aprendizaje de Preferencias",
                "policy_training": "Entrenamiento de Política",
                "evaluation": "Evaluación",
                "autonomous_sdlc": "SDLC Autónomo",
                "quality_gates": "Puertas de Calidad",
                "deployment": "Despliegue",
                "monitoring": "Monitoreo",
                "security_scan": "Escaneo de Seguridad",
                "performance_test": "Prueba de Rendimiento",
                "compliance_check": "Verificación de Cumplimiento"
            },
            SupportedLocale.FR_FR: {
                "welcome": "Bienvenue dans Robo-RLHF-Multimodal",
                "training_started": "Formation commencée",
                "training_completed": "Formation terminée avec succès",
                "error_occurred": "Une erreur s'est produite",
                "data_collection": "Collecte de Données",
                "preference_learning": "Apprentissage des Préférences",
                "policy_training": "Formation de Politique",
                "evaluation": "Évaluation",
                "autonomous_sdlc": "SDLC Autonome",
                "quality_gates": "Portes de Qualité",
                "deployment": "Déploiement",
                "monitoring": "Surveillance",
                "security_scan": "Analyse de Sécurité",
                "performance_test": "Test de Performance",
                "compliance_check": "Vérification de Conformité"
            },
            SupportedLocale.DE_DE: {
                "welcome": "Willkommen bei Robo-RLHF-Multimodal",
                "training_started": "Training gestartet",
                "training_completed": "Training erfolgreich abgeschlossen",
                "error_occurred": "Ein Fehler ist aufgetreten",
                "data_collection": "Datensammlung",
                "preference_learning": "Präferenzlernen",
                "policy_training": "Richtlinientraining",
                "evaluation": "Bewertung",
                "autonomous_sdlc": "Autonomer SDLC",
                "quality_gates": "Qualitätstüren",
                "deployment": "Bereitstellung",
                "monitoring": "Überwachung",
                "security_scan": "Sicherheitsscan",
                "performance_test": "Leistungstest",
                "compliance_check": "Compliance-Prüfung"
            },
            SupportedLocale.JA_JP: {
                "welcome": "Robo-RLHF-Multimodalへようこそ",
                "training_started": "トレーニングが開始されました",
                "training_completed": "トレーニングが正常に完了しました",
                "error_occurred": "エラーが発生しました",
                "data_collection": "データ収集",
                "preference_learning": "好み学習",
                "policy_training": "ポリシートレーニング",
                "evaluation": "評価",
                "autonomous_sdlc": "自律SDLC",
                "quality_gates": "品質ゲート",
                "deployment": "デプロイメント",
                "monitoring": "監視",
                "security_scan": "セキュリティスキャン",
                "performance_test": "パフォーマンステスト",
                "compliance_check": "コンプライアンスチェック"
            },
            SupportedLocale.ZH_CN: {
                "welcome": "欢迎使用 Robo-RLHF-Multimodal",
                "training_started": "训练已开始",
                "training_completed": "训练成功完成",
                "error_occurred": "发生错误",
                "data_collection": "数据收集",
                "preference_learning": "偏好学习",
                "policy_training": "策略训练",
                "evaluation": "评估",
                "autonomous_sdlc": "自主SDLC",
                "quality_gates": "质量门",
                "deployment": "部署",
                "monitoring": "监控",
                "security_scan": "安全扫描",
                "performance_test": "性能测试",
                "compliance_check": "合规检查"
            },
            SupportedLocale.PT_BR: {
                "welcome": "Bem-vindo ao Robo-RLHF-Multimodal",
                "training_started": "Treinamento iniciado",
                "training_completed": "Treinamento concluído com sucesso",
                "error_occurred": "Ocorreu um erro",
                "data_collection": "Coleta de Dados",
                "preference_learning": "Aprendizado de Preferências",
                "policy_training": "Treinamento de Política",
                "evaluation": "Avaliação",
                "autonomous_sdlc": "SDLC Autônomo",
                "quality_gates": "Portões de Qualidade",
                "deployment": "Implantação",
                "monitoring": "Monitoramento",
                "security_scan": "Varredura de Segurança",
                "performance_test": "Teste de Performance",
                "compliance_check": "Verificação de Conformidade"
            }
        }
        
        # Add default translations for missing locales
        for locale in SupportedLocale:
            if locale not in translations:
                translations[locale] = translations[SupportedLocale.EN_US].copy()
        
        return translations
    
    def _initialize_compliance_requirements(self) -> List[ComplianceRequirement]:
        """Initialize compliance requirements for different regions."""
        return [
            # GDPR (EU)
            ComplianceRequirement(
                region=ComplianceRegion.EU,
                regulation_name="GDPR",
                requirement_id="GDPR-001",
                description="Right to be forgotten - data deletion capabilities",
                data_types=[DataClassification.PII, DataClassification.SENSITIVE_PII],
                mandatory=True,
                implementation_status="implemented"
            ),
            ComplianceRequirement(
                region=ComplianceRegion.EU,
                regulation_name="GDPR",
                requirement_id="GDPR-002",
                description="Data portability - export user data in machine-readable format",
                data_types=[DataClassification.PII],
                mandatory=True,
                implementation_status="implemented"
            ),
            ComplianceRequirement(
                region=ComplianceRegion.EU,
                regulation_name="GDPR",
                requirement_id="GDPR-003",
                description="Consent management - explicit consent for data processing",
                data_types=[DataClassification.PII, DataClassification.SENSITIVE_PII],
                mandatory=True,
                implementation_status="implemented"
            ),
            
            # CCPA (California, US)
            ComplianceRequirement(
                region=ComplianceRegion.US,
                regulation_name="CCPA",
                requirement_id="CCPA-001",
                description="Consumer right to know about personal information collected",
                data_types=[DataClassification.PII],
                mandatory=True,
                implementation_status="implemented"
            ),
            ComplianceRequirement(
                region=ComplianceRegion.US,
                regulation_name="CCPA",
                requirement_id="CCPA-002",
                description="Consumer right to delete personal information",
                data_types=[DataClassification.PII, DataClassification.SENSITIVE_PII],
                mandatory=True,
                implementation_status="implemented"
            ),
            
            # LGPD (Brazil)
            ComplianceRequirement(
                region=ComplianceRegion.BRAZIL,
                regulation_name="LGPD",
                requirement_id="LGPD-001",
                description="Data subject rights - access, rectification, deletion",
                data_types=[DataClassification.PII],
                mandatory=True,
                implementation_status="planned"
            ),
            
            # Global AI Standards
            ComplianceRequirement(
                region=ComplianceRegion.GLOBAL,
                regulation_name="AI_ETHICS",
                requirement_id="AI-001",
                description="AI model transparency and explainability",
                data_types=[DataClassification.PUBLIC, DataClassification.INTERNAL],
                mandatory=False,
                implementation_status="implemented"
            ),
            ComplianceRequirement(
                region=ComplianceRegion.GLOBAL,
                regulation_name="AI_ETHICS",
                requirement_id="AI-002",
                description="Bias detection and mitigation in AI models",
                data_types=[DataClassification.PUBLIC],
                mandatory=False,
                implementation_status="implemented"
            )
        ]
    
    def _initialize_regional_settings(self) -> Dict[ComplianceRegion, Dict[str, Any]]:
        """Initialize regional-specific settings and configurations."""
        return {
            ComplianceRegion.EU: {
                "data_residency": "eu-central",
                "encryption_requirements": "AES-256",
                "audit_retention_days": 2555,  # 7 years
                "privacy_controls": {
                    "consent_required": True,
                    "opt_in_default": False,
                    "right_to_deletion": True,
                    "data_portability": True
                },
                "local_representatives": True,
                "dpo_required": True  # Data Protection Officer
            },
            ComplianceRegion.US: {
                "data_residency": "us-east",
                "encryption_requirements": "AES-256",
                "audit_retention_days": 2555,
                "privacy_controls": {
                    "consent_required": True,
                    "opt_in_default": True,
                    "right_to_deletion": True,
                    "data_portability": False
                },
                "local_representatives": False,
                "sox_compliance": True
            },
            ComplianceRegion.APAC: {
                "data_residency": "ap-southeast",
                "encryption_requirements": "AES-256",
                "audit_retention_days": 1825,  # 5 years
                "privacy_controls": {
                    "consent_required": True,
                    "opt_in_default": True,
                    "right_to_deletion": False,
                    "data_portability": False
                },
                "cross_border_restrictions": True
            },
            ComplianceRegion.CANADA: {
                "data_residency": "ca-central",
                "encryption_requirements": "AES-256",
                "audit_retention_days": 2190,  # 6 years
                "privacy_controls": {
                    "consent_required": True,
                    "opt_in_default": True,
                    "right_to_deletion": True,
                    "data_portability": True
                },
                "pipeda_compliance": True
            },
            ComplianceRegion.BRAZIL: {
                "data_residency": "sa-east",
                "encryption_requirements": "AES-256",
                "audit_retention_days": 1825,
                "privacy_controls": {
                    "consent_required": True,
                    "opt_in_default": False,
                    "right_to_deletion": True,
                    "data_portability": True
                },
                "lgpd_compliance": True
            }
        }
    
    def set_locale(self, locale: SupportedLocale) -> None:
        """Set the current locale for the application."""
        if locale not in self.locale_configs:
            self.logger.warning(f"Unsupported locale: {locale}, falling back to {self.default_locale}")
            locale = self.default_locale
        
        self.current_locale = locale
        config = self.locale_configs[locale]
        
        # Set system locale if available
        try:
            system_locale.setlocale(system_locale.LC_ALL, f"{config.language_code}_{config.country_code}.UTF-8")
        except system_locale.Error:
            self.logger.warning(f"System locale {config.language_code}_{config.country_code} not available")
        
        self.logger.info(f"🌍 Locale set to: {locale.value}")
    
    def translate(self, key: str, locale: Optional[SupportedLocale] = None) -> str:
        """
        Translate a text key to the specified or current locale.
        
        Args:
            key: Translation key
            locale: Target locale (uses current locale if None)
            
        Returns:
            Translated text or original key if translation not found
        """
        target_locale = locale or self.current_locale
        
        if target_locale in self.translations and key in self.translations[target_locale]:
            return self.translations[target_locale][key]
        
        # Fallback to English if translation not found
        if (target_locale != SupportedLocale.EN_US and 
            SupportedLocale.EN_US in self.translations and 
            key in self.translations[SupportedLocale.EN_US]):
            return self.translations[SupportedLocale.EN_US][key]
        
        # Return key if no translation found
        self.logger.warning(f"Translation not found for key '{key}' in locale '{target_locale}'")
        return key
    
    def format_date(self, dt: datetime, locale: Optional[SupportedLocale] = None) -> str:
        """Format datetime according to locale-specific format."""
        target_locale = locale or self.current_locale
        config = self.locale_configs.get(target_locale, self.locale_configs[self.default_locale])
        
        try:
            return dt.strftime(config.date_format)
        except ValueError:
            # Fallback to ISO format
            return dt.strftime("%Y-%m-%d")
    
    def format_time(self, dt: datetime, locale: Optional[SupportedLocale] = None) -> str:
        """Format time according to locale-specific format."""
        target_locale = locale or self.current_locale
        config = self.locale_configs.get(target_locale, self.locale_configs[self.default_locale])
        
        try:
            return dt.strftime(config.time_format)
        except ValueError:
            # Fallback to 24-hour format
            return dt.strftime("%H:%M")
    
    def format_number(self, number: Union[int, float], locale: Optional[SupportedLocale] = None) -> str:
        """Format number according to locale-specific format."""
        target_locale = locale or self.current_locale
        config = self.locale_configs.get(target_locale, self.locale_configs[self.default_locale])
        
        # Simple number formatting based on locale patterns
        if target_locale in [SupportedLocale.EN_US, SupportedLocale.ZH_CN]:
            # US/Chinese format: 1,234.56
            return f"{number:,.2f}" if isinstance(number, float) else f"{number:,}"
        elif target_locale in [SupportedLocale.ES_ES, SupportedLocale.DE_DE, SupportedLocale.PT_BR]:
            # European format: 1.234,56
            formatted = f"{number:,.2f}" if isinstance(number, float) else f"{number:,}"
            return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
        elif target_locale == SupportedLocale.FR_FR:
            # French format: 1 234,56
            formatted = f"{number:,.2f}" if isinstance(number, float) else f"{number:,}"
            return formatted.replace(",", " ").replace(".", ",")
        elif target_locale == SupportedLocale.JA_JP:
            # Japanese format: no decimals for currency
            return f"{int(number):,}" if isinstance(number, float) else f"{number:,}"
        else:
            # Default format
            return str(number)
    
    def format_currency(self, amount: Union[int, float], locale: Optional[SupportedLocale] = None) -> str:
        """Format currency according to locale-specific format."""
        target_locale = locale or self.current_locale
        config = self.locale_configs.get(target_locale, self.locale_configs[self.default_locale])
        
        formatted_number = self.format_number(amount, locale)
        
        # Currency symbol placement varies by locale
        if target_locale == SupportedLocale.EN_US:
            return f"${formatted_number}"
        elif target_locale in [SupportedLocale.ES_ES, SupportedLocale.FR_FR, SupportedLocale.DE_DE]:
            return f"{formatted_number} €"
        elif target_locale == SupportedLocale.JA_JP:
            return f"¥{formatted_number}"
        elif target_locale == SupportedLocale.ZH_CN:
            return f"¥{formatted_number}"
        elif target_locale == SupportedLocale.PT_BR:
            return f"R$ {formatted_number}"
        else:
            return f"{config.currency} {formatted_number}"
    
    def get_compliance_requirements(
        self, 
        region: Optional[ComplianceRegion] = None,
        data_type: Optional[DataClassification] = None
    ) -> List[ComplianceRequirement]:
        """Get compliance requirements filtered by region and/or data type."""
        requirements = self.compliance_requirements
        
        if region:
            requirements = [req for req in requirements if req.region == region or req.region == ComplianceRegion.GLOBAL]
        
        if data_type:
            requirements = [req for req in requirements if data_type in req.data_types]
        
        return requirements
    
    def validate_compliance(self, data_types: List[DataClassification], regions: List[ComplianceRegion]) -> Dict[str, Any]:
        """Validate compliance for specified data types and regions."""
        validation_results = {
            "overall_compliant": True,
            "region_compliance": {},
            "missing_requirements": [],
            "warnings": []
        }
        
        for region in regions:
            region_requirements = self.get_compliance_requirements(region)
            region_compliant = True
            missing_reqs = []
            
            for req in region_requirements:
                # Check if requirement applies to any of the data types
                if any(dt in req.data_types for dt in data_types):
                    if req.mandatory and req.implementation_status not in ["implemented", "compliant"]:
                        region_compliant = False
                        missing_reqs.append(req)
                        validation_results["overall_compliant"] = False
            
            validation_results["region_compliance"][region.value] = {
                "compliant": region_compliant,
                "total_requirements": len(region_requirements),
                "missing_requirements": len(missing_reqs)
            }
            validation_results["missing_requirements"].extend(missing_reqs)
        
        return validation_results
    
    def get_regional_settings(self, region: ComplianceRegion) -> Dict[str, Any]:
        """Get regional settings for deployment configuration."""
        return self.regional_settings.get(region, {})
    
    def generate_localization_report(self) -> Dict[str, Any]:
        """Generate comprehensive localization and compliance report."""
        report = {
            "current_locale": self.current_locale.value,
            "supported_locales": [locale.value for locale in self.locale_configs.keys()],
            "translation_coverage": {},
            "compliance_status": {},
            "regional_configurations": {},
            "recommendations": []
        }
        
        # Calculate translation coverage
        en_keys = set(self.translations[SupportedLocale.EN_US].keys())
        for locale, translations in self.translations.items():
            coverage = len(translations) / len(en_keys) * 100
            report["translation_coverage"][locale.value] = {
                "coverage_percentage": coverage,
                "total_keys": len(en_keys),
                "translated_keys": len(translations),
                "missing_keys": list(en_keys - set(translations.keys()))
            }
            
            if coverage < 100:
                report["recommendations"].append(
                    f"Complete translations for {locale.value} (currently {coverage:.1f}%)"
                )
        
        # Compliance status by region
        for region in ComplianceRegion:
            requirements = self.get_compliance_requirements(region)
            implemented = sum(1 for req in requirements if req.implementation_status == "implemented")
            total = len(requirements)
            
            report["compliance_status"][region.value] = {
                "total_requirements": total,
                "implemented_requirements": implemented,
                "compliance_percentage": (implemented / total * 100) if total > 0 else 100,
                "pending_requirements": [
                    req.requirement_id for req in requirements 
                    if req.implementation_status != "implemented"
                ]
            }
        
        # Regional configurations
        for region, settings in self.regional_settings.items():
            report["regional_configurations"][region.value] = {
                "data_residency": settings.get("data_residency"),
                "encryption_requirements": settings.get("encryption_requirements"),
                "privacy_controls": settings.get("privacy_controls", {})
            }
        
        return report


async def main():
    """Demonstrate global-first implementation features."""
    print("🌍 Global-First Implementation - Internationalization & Compliance")
    print("=" * 70)
    
    # Initialize globalization manager
    global_mgr = GlobalizationManager(default_locale=SupportedLocale.EN_US)
    
    # Demonstrate localization features
    print("\n🗣️ Localization Features")
    print("-" * 40)
    
    # Test different locales
    test_locales = [
        SupportedLocale.EN_US,
        SupportedLocale.ES_ES,
        SupportedLocale.FR_FR,
        SupportedLocale.DE_DE,
        SupportedLocale.JA_JP,
        SupportedLocale.ZH_CN
    ]
    
    for locale in test_locales:
        global_mgr.set_locale(locale)
        welcome_msg = global_mgr.translate("welcome")
        training_msg = global_mgr.translate("training_completed")
        
        # Format date, time, and currency
        now = datetime.now()
        formatted_date = global_mgr.format_date(now)
        formatted_time = global_mgr.format_time(now)
        formatted_currency = global_mgr.format_currency(1234.56)
        
        print(f"📍 {locale.value}:")
        print(f"   Welcome: {welcome_msg}")
        print(f"   Training: {training_msg}")
        print(f"   Date: {formatted_date} | Time: {formatted_time} | Currency: {formatted_currency}")
        print()
    
    # Demonstrate compliance features
    print("\n🛡️ Compliance Features")
    print("-" * 40)
    
    # Test compliance validation for different scenarios
    test_scenarios = [
        {
            "name": "EU Deployment with PII",
            "data_types": [DataClassification.PII, DataClassification.SENSITIVE_PII],
            "regions": [ComplianceRegion.EU]
        },
        {
            "name": "US Deployment with User Data",
            "data_types": [DataClassification.PII],
            "regions": [ComplianceRegion.US]
        },
        {
            "name": "Global AI System",
            "data_types": [DataClassification.PUBLIC, DataClassification.INTERNAL],
            "regions": [ComplianceRegion.GLOBAL, ComplianceRegion.EU, ComplianceRegion.US]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"📋 {scenario['name']}:")
        validation = global_mgr.validate_compliance(
            data_types=scenario['data_types'],
            regions=scenario['regions']
        )
        
        overall_status = "✅ COMPLIANT" if validation['overall_compliant'] else "❌ NON-COMPLIANT"
        print(f"   Status: {overall_status}")
        
        for region_name, region_data in validation['region_compliance'].items():
            region_status = "✅" if region_data['compliant'] else "❌"
            print(f"   {region_status} {region_name.upper()}: {region_data['total_requirements']} requirements, {region_data['missing_requirements']} missing")
        
        if validation['missing_requirements']:
            print(f"   Missing: {len(validation['missing_requirements'])} requirements need implementation")
        print()
    
    # Show regional settings
    print("\n🌐 Regional Settings")
    print("-" * 40)
    
    for region in [ComplianceRegion.EU, ComplianceRegion.US, ComplianceRegion.APAC]:
        settings = global_mgr.get_regional_settings(region)
        print(f"📍 {region.value.upper()}:")
        print(f"   Data Residency: {settings.get('data_residency', 'Not specified')}")
        print(f"   Encryption: {settings.get('encryption_requirements', 'Standard')}")
        privacy_controls = settings.get('privacy_controls', {})
        print(f"   Privacy Controls: Consent Required: {privacy_controls.get('consent_required', False)}")
        print(f"                    Right to Deletion: {privacy_controls.get('right_to_deletion', False)}")
        print()
    
    # Generate comprehensive report
    print("\n📊 Comprehensive Localization Report")
    print("-" * 40)
    
    report = global_mgr.generate_localization_report()
    
    print(f"Current Locale: {report['current_locale']}")
    print(f"Supported Locales: {len(report['supported_locales'])}")
    
    # Show translation coverage
    print("\n🗣️ Translation Coverage:")
    for locale, coverage_data in report['translation_coverage'].items():
        coverage_pct = coverage_data['coverage_percentage']
        status_emoji = "✅" if coverage_pct >= 100 else "⚠️" if coverage_pct >= 80 else "❌"
        print(f"   {status_emoji} {locale}: {coverage_pct:.1f}% ({coverage_data['translated_keys']}/{coverage_data['total_keys']})")
    
    # Show compliance status
    print("\n🛡️ Compliance Status:")
    for region, compliance_data in report['compliance_status'].items():
        compliance_pct = compliance_data['compliance_percentage']
        status_emoji = "✅" if compliance_pct >= 100 else "⚠️" if compliance_pct >= 80 else "❌"
        print(f"   {status_emoji} {region.upper()}: {compliance_pct:.1f}% ({compliance_data['implemented_requirements']}/{compliance_data['total_requirements']})")
    
    # Show recommendations
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
    
    # Save report
    report_file = f"global_first_report_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed report saved to: {report_file}")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())