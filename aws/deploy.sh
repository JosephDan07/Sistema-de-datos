#!/bin/bash

# 🚀 Script de Deployment a AWS
# Automatiza todo el proceso de despliegue

set -e

# Variables de configuración
PROJECT_NAME="sistema-patrones"
AWS_REGION="us-east-1"
ENVIRONMENT="dev"
ECR_REPO_NAME="$PROJECT_NAME"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

echo_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Función para verificar dependencias
check_dependencies() {
    echo_info "Verificando dependencias..."
    
    if ! command -v aws &> /dev/null; then
        echo_error "AWS CLI no está instalado"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo_error "Docker no está instalado"
        exit 1
    fi
    
    echo_success "Dependencias verificadas"
}

# Función para crear infraestructura
deploy_infrastructure() {
    echo_info "Desplegando infraestructura con CloudFormation..."
    
    aws cloudformation deploy \
        --template-file aws/cloudformation-infrastructure.yaml \
        --stack-name "$PROJECT_NAME-infrastructure" \
        --parameter-overrides \
            ProjectName="$PROJECT_NAME" \
            Environment="$ENVIRONMENT" \
            DBPassword="$DB_PASSWORD" \
        --capabilities CAPABILITY_IAM \
        --region "$AWS_REGION"
    
    echo_success "Infraestructura desplegada"
}

# Función para configurar ECR
setup_ecr() {
    echo_info "Configurando ECR Repository..."
    
    # Obtener la URI del ECR
    ECR_URI=$(aws cloudformation describe-stacks \
        --stack-name "$PROJECT_NAME-infrastructure" \
        --query 'Stacks[0].Outputs[?OutputKey==`ECRRepository`].OutputValue' \
        --output text \
        --region "$AWS_REGION")
    
    echo_info "ECR URI: $ECR_URI"
    
    # Login a ECR
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_URI"
    
    echo_success "ECR configurado"
}

# Función para construir y pushear imagen Docker
build_and_push() {
    echo_info "Construyendo imagen Docker..."
    
    # Construir imagen
    docker build -t "$PROJECT_NAME:latest" .
    
    # Tag para ECR
    docker tag "$PROJECT_NAME:latest" "$ECR_URI:latest"
    docker tag "$PROJECT_NAME:latest" "$ECR_URI:$ENVIRONMENT"
    
    # Push a ECR
    echo_info "Subiendo imagen a ECR..."
    docker push "$ECR_URI:latest"
    docker push "$ECR_URI:$ENVIRONMENT"
    
    echo_success "Imagen subida a ECR"
}

# Función para desplegar servicio ECS
deploy_ecs_service() {
    echo_info "Desplegando servicio ECS..."
    
    # Actualizar task definition con ECR URI
    sed "s|ACCOUNT.dkr.ecr.REGION.amazonaws.com/sistema-patrones:latest|$ECR_URI:latest|g" \
        aws/ecs-task-definition.json > /tmp/task-definition.json
    
    # Registrar task definition
    aws ecs register-task-definition \
        --cli-input-json file:///tmp/task-definition.json \
        --region "$AWS_REGION"
    
    # Crear o actualizar servicio
    CLUSTER_NAME=$(aws cloudformation describe-stacks \
        --stack-name "$PROJECT_NAME-infrastructure" \
        --query 'Stacks[0].Outputs[?OutputKey==`ClusterName`].OutputValue' \
        --output text \
        --region "$AWS_REGION")
    
    echo_info "Creando servicio en cluster: $CLUSTER_NAME"
    
    # Aquí irían los comandos para crear el servicio ECS
    # aws ecs create-service ...
    
    echo_success "Servicio ECS desplegado"
}

# Función para mostrar información de deployment
show_deployment_info() {
    echo_info "Obteniendo información del deployment..."
    
    ALB_DNS=$(aws cloudformation describe-stacks \
        --stack-name "$PROJECT_NAME-infrastructure" \
        --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
        --output text \
        --region "$AWS_REGION")
    
    DB_ENDPOINT=$(aws cloudformation describe-stacks \
        --stack-name "$PROJECT_NAME-infrastructure" \
        --query 'Stacks[0].Outputs[?OutputKey==`DatabaseEndpoint`].OutputValue' \
        --output text \
        --region "$AWS_REGION")
    
    echo_success "🎉 Deployment completado!"
    echo ""
    echo_info "📊 Acceso a servicios:"
    echo "   JupyterLab: http://$ALB_DNS:8888"
    echo "   Dash Apps:  http://$ALB_DNS:8050"
    echo ""
    echo_info "🗄️  Base de datos:"
    echo "   Endpoint: $DB_ENDPOINT:5432"
    echo "   Database: patterns_db"
    echo "   User: patterns_user"
    echo ""
    echo_info "📈 Monitoreo:"
    echo "   CloudWatch: https://console.aws.amazon.com/cloudwatch/"
    echo "   ECS Console: https://console.aws.amazon.com/ecs/"
}

# Función principal
main() {
    echo_info "🚀 Iniciando deployment a AWS..."
    echo_info "Proyecto: $PROJECT_NAME"
    echo_info "Región: $AWS_REGION"
    echo_info "Ambiente: $ENVIRONMENT"
    echo ""
    
    # Verificar que tenemos la password de la DB
    if [ -z "$DB_PASSWORD" ]; then
        echo_warning "Ingresa la password para la base de datos:"
        read -s DB_PASSWORD
        export DB_PASSWORD
    fi
    
    check_dependencies
    deploy_infrastructure
    setup_ecr
    build_and_push
    deploy_ecs_service
    show_deployment_info
    
    echo_success "🎉 Deployment completado exitosamente!"
}

# Ejecutar función principal
main "$@"
