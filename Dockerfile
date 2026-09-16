# Imagem base: Python 3.12 mínimo (sem compiladores nem ferramentas extras)
# Evitamos Alpine por incompatibilidade com numpy/sklearn (C extensions + glibc)
FROM python:3.12-slim

# Diretório de trabalho dentro do container
# Todos os caminhos relativos a partir daqui
WORKDIR /app

# Instala o uv (gerenciador de dependências e ambientes Python)
RUN pip install uv

# Copia só os arquivos de dependência primeiro — camada separada do código
# Estratégia: o que muda menos fica nas camadas inferiores para aproveitar o cache
# Se o código mudar mas o pyproject.toml não, essa camada é reutilizada
COPY pyproject.toml uv.lock ./

# Instala as dependências da aplicação
# --frozen: usa o lock file exato, sem resolver versões novamente
# --no-dev: exclui ferramentas de desenvolvimento (ruff etc.) da imagem de produção
RUN uv sync --frozen --no-dev

# Copia o código-fonte e o modelo treinado
# Ficam depois das dependências para não invalidar o cache de instalação
COPY src/ src/
COPY models/ models/

# Documenta que a aplicação escuta na porta 8000
# Não abre a porta — o mapeamento real é feito no docker run com -p
EXPOSE 8000

# Comando de inicialização do container
# --host 0.0.0.0: escuta em todas as interfaces (necessário para aceitar tráfego externo ao container)
# --port 8000: porta interna que será mapeada no docker run
CMD ["uv", "run", "uvicorn", "medical_triage.main:app", "--host", "0.0.0.0", "--port", "8000"]