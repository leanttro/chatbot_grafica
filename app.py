import os
import psycopg2
import psycopg2.extras
from flask import Flask, jsonify, request, render_template
from dotenv import load_dotenv
from flask_cors import CORS
import json
import google.generativeai as genai
import decimal
import traceback # Para logs de erro mais detalhados

# --- CONFIGURAÇÃO INICIAL ---
load_dotenv() # Carrega variáveis do arquivo .env (DATABASE_URL, GEMINI_API_KEY)
app = Flask(__name__, template_folder='.', static_folder='.') # Servir arquivos da pasta raiz
CORS(app)

# Configura o Gemini (lê a chave do .env)
try:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ ERRO CRÍTICO: Variável de ambiente GEMINI_API_KEY não encontrada no arquivo .env.")
    else:
        genai.configure(api_key=api_key)
        print("✅ API Key do Gemini configurada.")
except Exception as e:
    print(f"❌ Erro ao configurar a API do Gemini: {e}")

# Conexão com o DB (lê a URL do .env)
def get_db_connection():
    """Cria e retorna uma conexão com o banco de dados PostgreSQL."""
    conn = None
    try:
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            print("❌ ERRO CRÍTICO: Variável de ambiente DATABASE_URL não encontrada no arquivo .env.")
            raise ValueError("DATABASE_URL não definida")
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: Não foi possível conectar ao banco de dados: {e}")
        raise

# Helper para formatar dados (decimal, etc.) - Você já tem isso
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            # Converte Decimal para string para evitar problemas de precisão com float
            return str(obj)
        # Permite que a classe base lide com outros tipos
        return super(DecimalEncoder, self).default(obj)
app.json_encoder = DecimalEncoder

# --- LÓGICA DO CHATBOT (RAG com tabela 'grafica') ---

def get_grafica_data_for_bot(limit=50):
    """Busca os últimos 'limit' registros da tabela 'grafica' para o bot."""
    conn = None
    try:
        conn = get_db_connection()
        # Usando RealDictCursor para obter resultados como dicionários
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Seleciona as colunas relevantes e ordena pelos IDs mais recentes
        cur.execute(f"SELECT id, quantidade, produto, material, impressao, valor_final FROM grafica ORDER BY id DESC LIMIT {limit}")
        registros_raw = cur.fetchall()
        cur.close()

        # Converte para lista de dicionários padrão
        registros = [dict(row) for row in registros_raw]
        print(f"ℹ️ Dados do Bot: Carregados {len(registros)} registros recentes da tabela 'grafica'.")
        return registros
    except psycopg2.errors.UndefinedTable:
         print(f"⚠️ AVISO: A tabela 'grafica' não foi encontrada. O chatbot não terá contexto de pedidos.")
         return [] # Retorna lista vazia se a tabela não existe
    except Exception as e:
        print(f"❌ ERRO ao buscar dados da tabela 'grafica' para o bot: {e}")
        traceback.print_exc()
        return [] # Retorna lista vazia em caso de erro
    finally:
        if conn: conn.close()

# Carrega os dados UMA VEZ na inicialização do servidor
grafica_data_context = get_grafica_data_for_bot()

# Converte os dados para JSON para injetar no prompt
# Usa o DecimalEncoder customizado e garante que caracteres PT-BR fiquem corretos
grafica_json_context = json.dumps(grafica_data_context, cls=DecimalEncoder, ensure_ascii=False, separators=(',', ':'))

# Define o Prompt do Sistema para o Gemini
SYSTEM_PROMPT = f"""
Você é o 'GrafiBot', um assistente técnico interno de uma gráfica rápida, especializado em orçamentos e consulta de pedidos recentes.
Sua única fonte de verdade é a base de dados de pedidos recentes em JSON fornecida abaixo.

--- BASE DE DADOS (Pedidos Recentes - JSON) ---
{grafica_json_context}
--- FIM DA BASE DE DADOS ---

REGRAS ESTRITAS:
1.  **FOCO NOS DADOS:** Baseie 100% das suas respostas nos dados JSON fornecidos. Use os nomes, quantidades e valores EXATOS da base.
2.  **NÃO ALUCINE:** Você NUNCA deve inventar um pedido, preço, material ou qualquer informação que não esteja na base. Se a informação não estiver lá, diga que não encontrou nos registros recentes.
3.  **SEJA UM CONSULTOR:** Aja como um consultor interno. Seja direto, técnico e preciso. Ex: "Encontrei o Pedido ID 123: 1000 Cartões de Visita em Couchê 300g, 4x4 Cores, por R$ XXX,XX."
4.  **PARA ORÇAMENTOS (SIMULADO):** Se o usuário pedir um orçamento (ex: "quanto custa 500 folders?"), use a BASE DE DADOS como *inspiração* para dar uma *estimativa*. Diga algo como: "Com base em pedidos recentes similares, um pedido de 500 folders [Material X], [Impressão Y] custou aproximadamente R$ ZZZ,ZZ. Para um orçamento exato, por favor, use o formulário de cadastro." **NUNCA** dê um preço exato se não estiver na base.
5.  **CONSULTA DE VENDAS:** Se o usuário perguntar sobre vendas recentes (ex: "quais foram os últimos pedidos de etiqueta?"), liste os pedidos relevantes da BASE DE DADOS.
6.  **RECUSE OUTROS ASSUNTOS:** Responda apenas sobre orçamentos e pedidos da gráfica. Recuse educadamente outros tópicos.
"""

# Inicializa o Modelo e a Sessão de Chat
model = None
chat_session = None
try:
    if api_key: # Só tenta inicializar se a API key foi carregada
        model = genai.GenerativeModel('gemini-flash-latest')
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": [SYSTEM_PROMPT]},
                {"role": "model", "parts": ["Entendido. Sou o GrafiBot. Minha base de pedidos recentes está carregada. Pronto para consultar ou estimar orçamentos."]}
            ]
        )
        print("✅ Modelo Gemini ('gemini-flash-latest') inicializado com o contexto da tabela 'grafica'.")
    else:
        print("⚠️ AVISO: API Key do Gemini não carregada. O chatbot não funcionará.")

except Exception as e:
    print(f"❌ ERRO CRÍTICO ao inicializar o GenerativeModel: {e}")
    traceback.print_exc()

# --- ROTAS DA APLICAÇÃO ---

# Rota para servir o HTML principal (seu index.html)
@app.route('/')
def index():
    """Serve a página principal."""
    # Flask procura automaticamente por 'index.html' na pasta raiz devido a `template_folder='.'`
    return render_template('index.html')

# Rota para a API do Chatbot
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Recebe a mensagem do usuário e retorna a resposta do Gemini."""
    if not model or not chat_session:
        print("❌ Erro na Rota /api/chat: Sessão do Gemini não inicializada.")
        return jsonify({'error': 'Serviço de chat indisponível no momento.'}), 503

    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({'error': 'Mensagem não pode ser vazia.'}), 400

        # Envia a mensagem para o Gemini (o histórico é mantido no 'chat_session')
        # Inclui configuração para tentar evitar bloqueios
        response = chat_session.send_message(
            user_message,
            generation_config=genai.types.GenerationConfig(temperature=0.7),
            safety_settings={'HATE': 'BLOCK_NONE', 'HARASSMENT': 'BLOCK_NONE',
                             'SEXUAL' : 'BLOCK_NONE', 'DANGEROUS' : 'BLOCK_NONE'}
        )

        # Retorna a resposta do modelo para o front-end
        print(f"🤖 Resposta do Bot: {response.text[:100]}...") # Log curto da resposta
        return jsonify({'reply': response.text})

    except genai.types.generation_types.StopCandidateException as stop_ex:
        print(f"⚠️ API BLOQUEOU a resposta por segurança: {stop_ex}")
        return jsonify({'reply': "Desculpe, não posso gerar uma resposta para essa solicitação específica. Posso ajudar com orçamentos ou consulta de pedidos?"})
    except Exception as e:
        print(f"❌ Erro ao chamar a API do Gemini: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Ocorreu um erro ao processar sua mensagem com a IA.'}), 503

# Rota NOVA para registrar o pedido na tabela 'grafica'
@app.route('/api/registrar_pedido', methods=['POST'])
def registrar_pedido():
    """Recebe dados do formulário HTML e insere na tabela 'grafica'."""
    dados = request.json
    conn = None
    print(f"ℹ️ Recebido POST em /api/registrar_pedido: {dados}") # Log para ver os dados chegando

    # Validação simples (pode ser melhorada)
    if not dados or 'quantidade' not in dados or 'produto' not in dados or 'valorFinal' not in dados:
         print("❌ Erro em /api/registrar_pedido: Dados incompletos recebidos.")
         return jsonify({'error': 'Dados incompletos para registrar o pedido.'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # SQL para inserir na tabela 'grafica' (ajuste as colunas se necessário)
        # Garante que os nomes aqui batam EXATAMENTE com os nomes das colunas no seu DB
        sql_insert = """
        INSERT INTO grafica (
            QUANTIDADE, PRODUTO, MATERIAL, IMPRESSAO, LARGURA, ALTURA,
            TIPO_DE_CORTE, ACABAMENTO, EXTRA, VALOR_FINAL
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        # Monta a tupla de valores na ordem correta, usando .get() para campos opcionais
        valores = (
            dados.get('quantidade'),
            dados.get('produto'),
            dados.get('material'),
            dados.get('impressao'),
            dados.get('largura') or None, # Trata campos numéricos vazios como NULL
            dados.get('altura') or None,
            dados.get('tipoCorte') or None, # Nome do JS é tipoCorte
            dados.get('acabamento'),
            dados.get('extra') or None, # Trata 'Nenhum' ou vazio como NULL
            dados.get('valorFinal') # Nome do JS é valorFinal
        )

        cur.execute(sql_insert, valores)
        conn.commit()
        cur.close()
        print("✅ Pedido registrado com sucesso na tabela 'grafica'.")

        # ATENÇÃO: Reiniciar o servidor ainda é necessário para o BOT ver o novo dado
        # Poderíamos recarregar o contexto aqui, mas o restart é mais simples para a demo.
        global grafica_data_context, grafica_json_context, SYSTEM_PROMPT, chat_session
        print("🔄 Recarregando contexto do bot após novo pedido...")
        grafica_data_context = get_grafica_data_for_bot()
        grafica_json_context = json.dumps(grafica_data_context, cls=DecimalEncoder, ensure_ascii=False, separators=(',', ':'))
        SYSTEM_PROMPT = f"""
        Você é o 'GrafiBot'...
        --- BASE DE DADOS (Pedidos Recentes - JSON) ---
        {grafica_json_context}
        --- FIM DA BASE DE DADOS ---
        ... (Resto do prompt) ...
        """
        # Reinicia a sessão de chat com o novo prompt
        if model:
             chat_session = model.start_chat(
                history=[
                    {"role": "user", "parts": [SYSTEM_PROMPT]},
                    {"role": "model", "parts": ["Entendido. Sou o GrafiBot. Base de pedidos atualizada. Pronto para ajudar."]}
                ]
            )
             print("✅ Contexto do bot atualizado com o novo pedido.")
        else:
             print("⚠️ Bot não inicializado, não foi possível atualizar contexto.")


        return jsonify({'success': 'Pedido registrado com sucesso! O chatbot agora pode ver este pedido.'}), 201

    except psycopg2.errors.UndefinedTable:
        print(f"❌ ERRO em /api/registrar_pedido: A tabela 'grafica' não existe.")
        return jsonify({'error': "Erro interno: Tabela 'grafica' não encontrada."}), 500
    except psycopg2.Error as db_err:
        print(f"❌ ERRO de Banco de Dados em /api/registrar_pedido: {db_err}")
        traceback.print_exc()
        if conn: conn.rollback()
        return jsonify({'error': 'Erro ao salvar o pedido no banco de dados.'}), 500
    except Exception as e:
        print(f"❌ ERRO inesperado em /api/registrar_pedido: {e}")
        traceback.print_exc()
        if conn: conn.rollback()
        return jsonify({'error': 'Erro interno do servidor ao registrar pedido.'}), 500
    finally:
        if conn: conn.close()

# --- Execução do App ---
if __name__ == '__main__':
    # Render usa a variável PORT, senão usa 5000 localmente
    port = int(os.environ.get("PORT", 5000))
    # debug=False é importante para produção no Render
    # use_reloader=False evita que o Render reinicie o worker constantemente
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
