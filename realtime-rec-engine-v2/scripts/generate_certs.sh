#!/bin/bash
# Certificate generation script for rec-engine TLS/mTLS
# This script generates self-signed certificates for development/testing
# For production, use cert-manager with Let's Encrypt or your organization's CA

set -euo pipefail

CERT_DIR="./certs"
CA_KEY="${CERT_DIR}/ca.key"
CA_CRT="${CERT_DIR}/ca.crt"
DAYS_VALID=365

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for required tools
check_tools() {
    for tool in openssl cfssl cfssljson; do
        if ! command -v "$tool" &> /dev/null; then
            log_warn "$tool not found. Install with: brew install cfssl (macOS) or download from https://github.com/cloudflare/cfssl"
        fi
    done
}

# Create CA
create_ca() {
    log_info "Creating Certificate Authority..."
    mkdir -p "${CERT_DIR}"

    if [[ -f "${CA_KEY}" && -f "${CA_CRT}" ]]; then
        log_warn "CA already exists, skipping..."
        return
    fi

    # Generate CA private key
    openssl genrsa -out "${CA_KEY}" 4096

    # Generate CA certificate
    openssl req -x509 -new -nodes -key "${CA_KEY}" -sha256 -days "${DAYS_VALID}" \
        -out "${CA_CRT}" \
        -subj "/CN=rec-engine-CA/O=rec-engine/OU=Security" \
        -addext "basicConstraints=critical,CA:true" \
        -addext "keyUsage=critical,keyCertSign,cRLSign"

    log_info "CA created: ${CA_CRT}"
}

# Generate server certificate
generate_server_cert() {
    local name=$1
    local cn=$2
    local sans=$3
    local key_file="${CERT_DIR}/${name}.key"
    local csr_file="${CERT_DIR}/${name}.csr"
    local crt_file="${CERT_DIR}/${name}.crt"

    log_info "Generating server certificate for ${cn}..."

    # Generate private key
    openssl genrsa -out "${key_file}" 2048

    # Generate CSR with SANs
    cat > "${CERT_DIR}/${name}.cnf" <<EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
CN = ${cn}
O = rec-engine
OU = Server

[v3_req]
subjectAltName = @alt_names
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth,clientAuth

[alt_names]
EOF

    # Add SANs
    IFS=',' read -ra SAN_ARRAY <<< "${sans}"
    i=1
    for san in "${SAN_ARRAY[@]}"; do
        if [[ $san =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "IP.${i} = ${san}" >> "${CERT_DIR}/${name}.cnf"
        else
            echo "DNS.${i} = ${san}" >> "${CERT_DIR}/${name}.cnf"
        fi
        ((i++))
    done

    # Generate CSR
    openssl req -new -key "${key_file}" -out "${csr_file}" -config "${CERT_DIR}/${name}.cnf"

    # Sign certificate with CA
    openssl x509 -req -in "${csr_file}" -CA "${CA_CRT}" -CAkey "${CA_KEY}" \
        -CAcreateserial -out "${crt_file}" -days "${DAYS_VALID}" -sha256 \
        -extensions v3_req -extfile "${CERT_DIR}/${name}.cnf"

    # Cleanup
    rm "${csr_file}" "${CERT_DIR}/${name}.cnf"

    log_info "Server certificate created: ${crt_file}"
}

# Generate client certificate
generate_client_cert() {
    local name=$1
    local cn=$2
    local key_file="${CERT_DIR}/${name}.key"
    local csr_file="${CERT_DIR}/${name}.csr"
    local crt_file="${CERT_DIR}/${name}.crt"

    log_info "Generating client certificate for ${cn}..."

    # Generate private key
    openssl genrsa -out "${key_file}" 2048

    # Generate CSR
    cat > "${CERT_DIR}/${name}.cnf" <<EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
CN = ${cn}
O = rec-engine
OU = Client

[v3_req]
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = clientAuth
EOF

    openssl req -new -key "${key_file}" -out "${csr_file}" -config "${CERT_DIR}/${name}.cnf" \
        -subj "/CN=${cn}/O=rec-engine/OU=Client"

    # Sign certificate with CA
    openssl x509 -req -in "${csr_file}" -CA "${CA_CRT}" -CAkey "${CA_KEY}" \
        -CAcreateserial -out "${crt_file}" -days "${DAYS_VALID}" -sha256 \
        -extensions v3_req -extfile "${CERT_DIR}/${name}.cnf"

    # Cleanup
    rm "${csr_file}" "${CERT_DIR}/${name}.cnf"

    log_info "Client certificate created: ${crt_file}"
}

# Generate Java keystore for Kafka/ZooKeeper
generate_keystore() {
    local name=$1
    local cn=$2
    local password=$3
    local jks_file="${CERT_DIR}/${name}-keystore.jks"

    log_info "Generating Java keystore for ${cn}..."

    # Create PKCS12 first
    openssl pkcs12 -export -in "${CERT_DIR}/${name}.crt" -inkey "${CERT_DIR}/${name}.key" \
        -out "${CERT_DIR}/${name}.p12" -name "${name}" \
        -password "pass:${password}" -CAfile "${CA_CRT}" -chain

    # Convert to JKS
    keytool -importkeystore -deststorepass "${password}" -destkeypass "${password}" \
        -destkeystore "${jks_file}" -srckeystore "${CERT_DIR}/${name}.p12" \
        -srcstoretype PKCS12 -srcstorepass "${password}" -alias "${name}" -noprompt

    # Create truststore
    local truststore_file="${CERT_DIR}/${name}-truststore.jks"
    keytool -importcert -file "${CA_CRT}" -keystore "${truststore_file}" \
        -storepass "${password}" -alias ca -noprompt

    # Cleanup
    rm "${CERT_DIR}/${name}.p12"

    log_info "Keystore created: ${jks_file}"
}

# Main
main() {
    log_info "Starting certificate generation for rec-engine..."

    check_tools
    create_ca

    # Server certificates
    generate_server_cert "api-server" "api.rec-engine.example.com" "api.rec-engine.example.com,api.rec-engine.local,localhost,127.0.0.1,rec-engine-api,rec-engine-api.rec-engine.svc.cluster.local"
    generate_server_cert "kafka-server" "kafka.rec-engine.example.com" "kafka.rec-engine.example.com,kafka,kafka.rec-engine.svc.cluster.local,kafka-0.kafka-headless,kafka-1.kafka-headless,kafka-2.kafka-headless,localhost,127.0.0.1"
    generate_server_cert "redis-server" "redis.rec-engine.example.com" "redis.rec-engine.example.com,redis,redis-master,redis-master.rec-engine.svc.cluster.local,localhost,127.0.0.1"
    generate_server_cert "postgres-server" "postgres.rec-engine.example.com" "postgres.rec-engine.example.com,postgres,postgres-primary,postgres-primary.rec-engine.svc.cluster.local,localhost,127.0.0.1"
    generate_server_cert "schema-registry-server" "schema-registry.rec-engine.example.com" "schema-registry.rec-engine.example.com,schema-registry,schema-registry.rec-engine.svc.cluster.local,localhost,127.0.0.1"
    generate_server_cert "grafana-server" "grafana.rec-engine.example.com" "grafana.rec-engine.example.com,grafana,grafana.rec-engine.svc.cluster.local,localhost,127.0.0.1"
    generate_server_cert "vault-server" "vault.rec-engine.example.com" "vault.rec-engine.example.com,vault,vault.rec-engine.svc.cluster.local,localhost,127.0.0.1"

    # Client certificates
    generate_client_cert "kafka-client" "kafka-client"
    generate_client_cert "redis-client" "redis-client"
    generate_client_cert "postgres-client" "postgres-client"
    generate_client_cert "api-client" "api-client"

    # Java keystores for Kafka/ZooKeeper/Schema Registry
    # Use a default password for development (override in production!)
    KEYSTORE_PASSWORD="changeit"
    generate_keystore "zookeeper" "zookeeper.rec-engine.example.com" "${KEYSTORE_PASSWORD}"
    generate_keystore "kafka" "kafka.rec-engine.example.com" "${KEYSTORE_PASSWORD}"
    generate_keystore "schema-registry" "schema-registry.rec-engine.example.com" "${KEYSTORE_PASSWORD}"

    # Create Kubernetes secret manifest
    log_info "Creating Kubernetes secret manifest..."
    cat > "${CERT_DIR}/k8s-secret.yaml" <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: rec-engine-certs
  namespace: rec-engine
type: Opaque
data:
  ca.crt: $(base64 -w0 "${CA_CRT}")
  api-server.crt: $(base64 -w0 "${CERT_DIR}/api-server.crt")
  api-server.key: $(base64 -w0 "${CERT_DIR}/api-server.key")
  kafka-client.crt: $(base64 -w0 "${CERT_DIR}/kafka-client.crt")
  kafka-client.key: $(base64 -w0 "${CERT_DIR}/kafka-client.key")
  redis-client.crt: $(base64 -w0 "${CERT_DIR}/redis-client.crt")
  redis-client.key: $(base64 -w0 "${CERT_DIR}/redis-client.key")
  postgres-client.crt: $(base64 -w0 "${CERT_DIR}/postgres-client.crt")
  postgres-client.key: $(base64 -w0 "${CERT_DIR}/postgres-client.key")
  # Java keystores
  zookeeper-keystore.jks: $(base64 -w0 "${CERT_DIR}/zookeeper-keystore.jks")
  zookeeper-truststore.jks: $(base64 -w0 "${CERT_DIR}/zookeeper-truststore.jks")
  kafka-keystore.jks: $(base64 -w0 "${CERT_DIR}/kafka-keystore.jks")
  kafka-truststore.jks: $(base64 -w0 "${CERT_DIR}/kafka-truststore.jks")
  schema-registry-keystore.jks: $(base64 -w0 "${CERT_DIR}/schema-registry-keystore.jks")
  schema-registry-truststore.jks: $(base64 -w0 "${CERT_DIR}/schema-registry-truststore.jks")
EOF

    log_info "Certificate generation complete!"
    log_info "Certificates stored in: ${CERT_DIR}"
    log_info "Kubernetes secret manifest: ${CERT_DIR}/k8s-secret.yaml"
    log_warn "IMPORTANT: For production, use cert-manager with Let's Encrypt or your organization's CA!"
    log_warn "Default keystore password: ${KEYSTORE_PASSWORD} - CHANGE IN PRODUCTION!"
}

main "$@"