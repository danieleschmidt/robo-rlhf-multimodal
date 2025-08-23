# Terraform configuration for production environment

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
resource "aws_vpc" "robo_rlhf_production" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "robo-rlhf-production-vpc"
    Environment = "production"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "robo_rlhf_production" {
  name     = "robo-rlhf-production"
  role_arn = aws_iam_role.eks_cluster.arn
  
  vpc_config {
    subnet_ids = aws_subnet.robo_rlhf_production[*].id
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
  ]
  
  tags = {
    Environment = "production"
  }
}

# EKS Node Group
resource "aws_eks_node_group" "robo_rlhf_production" {
  cluster_name    = aws_eks_cluster.robo_rlhf_production.name
  node_group_name = "robo-rlhf-production-nodes"
  node_role_arn   = aws_iam_role.eks_node_group.arn
  subnet_ids      = aws_subnet.robo_rlhf_production[*].id
  
  capacity_type  = "ON_DEMAND"
  instance_types = ["m5.large"]
  
  scaling_config {
    desired_size = 5
    max_size     = 50
    min_size     = 5
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
  
  tags = {
    Environment = "production"
  }
}

# RDS Database
resource "aws_db_instance" "robo_rlhf_production" {
  identifier     = "robo-rlhf-production-db"
  engine         = "postgres"
  engine_version = "14.9"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  
  db_name  = "robo_rlhf"
  username = "dbuser"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.robo_rlhf_production.name
  
  skip_final_snapshot = false
  
  tags = {
    Environment = "production"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "robo_rlhf_production" {
  name       = "robo-rlhf-production-cache-subnet"
  subnet_ids = aws_subnet.robo_rlhf_production[*].id
}

resource "aws_elasticache_cluster" "robo_rlhf_production" {
  cluster_id           = "robo-rlhf-production"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.robo_rlhf_production.name
  security_group_ids   = [aws_security_group.redis.id]
  
  tags = {
    Environment = "production"
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.robo_rlhf_production.endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.robo_rlhf_production.name
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.robo_rlhf_production.endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_cluster.robo_rlhf_production.cache_nodes[0].address
}
