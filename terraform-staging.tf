# Terraform configuration for staging environment

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
resource "aws_vpc" "robo_rlhf_staging" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "robo-rlhf-staging-vpc"
    Environment = "staging"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "robo_rlhf_staging" {
  name     = "robo-rlhf-staging"
  role_arn = aws_iam_role.eks_cluster.arn
  
  vpc_config {
    subnet_ids = aws_subnet.robo_rlhf_staging[*].id
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
  ]
  
  tags = {
    Environment = "staging"
  }
}

# EKS Node Group
resource "aws_eks_node_group" "robo_rlhf_staging" {
  cluster_name    = aws_eks_cluster.robo_rlhf_staging.name
  node_group_name = "robo-rlhf-staging-nodes"
  node_role_arn   = aws_iam_role.eks_node_group.arn
  subnet_ids      = aws_subnet.robo_rlhf_staging[*].id
  
  capacity_type  = "ON_DEMAND"
  instance_types = ["m5.large"]
  
  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 2
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
  
  tags = {
    Environment = "staging"
  }
}

# RDS Database
resource "aws_db_instance" "robo_rlhf_staging" {
  identifier     = "robo-rlhf-staging-db"
  engine         = "postgres"
  engine_version = "14.9"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  
  db_name  = "robo_rlhf"
  username = "dbuser"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.robo_rlhf_staging.name
  
  skip_final_snapshot = true
  
  tags = {
    Environment = "staging"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "robo_rlhf_staging" {
  name       = "robo-rlhf-staging-cache-subnet"
  subnet_ids = aws_subnet.robo_rlhf_staging[*].id
}

resource "aws_elasticache_cluster" "robo_rlhf_staging" {
  cluster_id           = "robo-rlhf-staging"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.robo_rlhf_staging.name
  security_group_ids   = [aws_security_group.redis.id]
  
  tags = {
    Environment = "staging"
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
  value       = aws_eks_cluster.robo_rlhf_staging.endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.robo_rlhf_staging.name
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.robo_rlhf_staging.endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_cluster.robo_rlhf_staging.cache_nodes[0].address
}
