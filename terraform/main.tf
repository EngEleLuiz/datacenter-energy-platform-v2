# =============================================================================
# infra/main.tf
# Datacenter Energy Intelligence Platform — AWS Infrastructure
# Provisions: S3 Data Lake, Glue Catalog, Athena, IAM roles
# =============================================================================

terraform {
  required_version = ">= 1.8"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to use S3 remote state (recommended for teams)
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "datacenter-platform/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# S3 — Medallion Data Lake (Bronze / Silver / Gold)
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "bronze" {
  bucket        = "${var.project_name}-bronze-${var.environment}"
  force_destroy = var.environment != "prod"

  tags = local.common_tags
}

resource "aws_s3_bucket" "silver" {
  bucket        = "${var.project_name}-silver-${var.environment}"
  force_destroy = var.environment != "prod"

  tags = local.common_tags
}

resource "aws_s3_bucket" "gold" {
  bucket        = "${var.project_name}-gold-${var.environment}"
  force_destroy = var.environment != "prod"

  tags = local.common_tags
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket        = "${var.project_name}-mlflow-${var.environment}"
  force_destroy = var.environment != "prod"

  tags = local.common_tags
}

# Versioning on Silver and Gold (important for data lineage)
resource "aws_s3_bucket_versioning" "silver" {
  bucket = aws_s3_bucket.silver.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_versioning" "gold" {
  bucket = aws_s3_bucket.gold.id
  versioning_configuration { status = "Enabled" }
}

# Lifecycle: Bronze expires after 30 days (raw data pruning)
resource "aws_s3_bucket_lifecycle_configuration" "bronze_lifecycle" {
  bucket = aws_s3_bucket.bronze.id

  rule {
    id     = "expire-raw-data"
    status = "Enabled"

    expiration {
      days = 30
    }

    filter { prefix = "" }
  }
}

# Block all public access on all buckets
resource "aws_s3_bucket_public_access_block" "all" {
  for_each = {
    bronze   = aws_s3_bucket.bronze.id
    silver   = aws_s3_bucket.silver.id
    gold     = aws_s3_bucket.gold.id
    mlflow   = aws_s3_bucket.mlflow_artifacts.id
  }

  bucket                  = each.value
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# AWS Glue — Data Catalog (metadata store for Athena queries)
# ---------------------------------------------------------------------------
resource "aws_glue_catalog_database" "bronze" {
  name        = "${replace(var.project_name, "-", "_")}_bronze"
  description = "Raw landing zone — unvalidated telemetry"
}

resource "aws_glue_catalog_database" "silver" {
  name        = "${replace(var.project_name, "-", "_")}_silver"
  description = "Validated, typed, partitioned data"
}

resource "aws_glue_catalog_database" "gold" {
  name        = "${replace(var.project_name, "-", "_")}_gold"
  description = "Aggregated KPIs and ML feature tables"
}

# Glue Crawler — auto-discovers schema from Silver Parquet files
# resource "aws_glue_crawler" "silver_crawler" {
#   name          = "${var.project_name}-silver-crawler"
#   role          = aws_iam_role.glue_role.arn
#   database_name = aws_glue_catalog_database.silver.name
#   schedule      = "cron(0/15 * * * ? *)"
#
#   s3_target {
#     path = "s3://${aws_s3_bucket.silver.bucket}/"
#   }
#
#   schema_change_policy {
#     delete_behavior = "LOG"
#     update_behavior = "UPDATE_IN_DATABASE"
#   }
#
#   tags = local.common_tags
# }

# ---------------------------------------------------------------------------
# Athena — serverless SQL on S3
# ---------------------------------------------------------------------------
resource "aws_athena_workgroup" "main" {
  name = var.project_name

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://${aws_s3_bucket.gold.bucket}/athena-results/"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }

    bytes_scanned_cutoff_per_query = 1073741824  # 1 GB safety limit
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# IAM — Roles and Policies
# ---------------------------------------------------------------------------
resource "aws_iam_role" "glue_role" {
  name = "${var.project_name}-glue-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "glue.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy" "glue_s3_access" {
  name = "${var.project_name}-glue-s3-policy"
  role = aws_iam_role.glue_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.bronze.arn, "${aws_s3_bucket.bronze.arn}/*",
          aws_s3_bucket.silver.arn, "${aws_s3_bucket.silver.arn}/*",
          aws_s3_bucket.gold.arn,   "${aws_s3_bucket.gold.arn}/*",
        ]
      }
    ]
  })
}

# Application role (for the Kafka consumer / Python pipeline running on EC2 or Lambda)
resource "aws_iam_role" "app_role" {
  name = "${var.project_name}-app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "app_s3_policy" {
  name = "${var.project_name}-app-s3-policy"
  role = aws_iam_role.app_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:*"]
        Resource = [
          aws_s3_bucket.bronze.arn,         "${aws_s3_bucket.bronze.arn}/*",
          aws_s3_bucket.silver.arn,         "${aws_s3_bucket.silver.arn}/*",
          aws_s3_bucket.gold.arn,           "${aws_s3_bucket.gold.arn}/*",
          aws_s3_bucket.mlflow_artifacts.arn,"${aws_s3_bucket.mlflow_artifacts.arn}/*",
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["glue:*", "athena:*"]
        Resource = ["*"]
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# CloudWatch — Log Groups for pipeline observability
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/datacenter-platform/pipeline"
  retention_in_days = 30

  tags = local.common_tags
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
output "bronze_bucket" { value = aws_s3_bucket.bronze.bucket }
output "silver_bucket" { value = aws_s3_bucket.silver.bucket }
output "gold_bucket"   { value = aws_s3_bucket.gold.bucket }
output "glue_silver_db" { value = aws_glue_catalog_database.silver.name }
output "athena_workgroup" { value = aws_athena_workgroup.main.name }
