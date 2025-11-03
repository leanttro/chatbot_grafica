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
# Servir arquivos estáticos (como logo.png) da pasta raiz '.'
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')
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

# Helper para formatar dados (decimal, etc.)
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)
app.json_encoder = DecimalEncoder

# --- LÓGICA DO CHATBOT (RAG com tabela 'grafica') ---

def get_grafica_data_for_bot(limit=50):
    """Busca os últimos 'limit' registros da tabela 'grafica' para o bot."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SELECT id, quantidade, produto, material, impressao, largura, altura, valor_final FROM grafica ORDER BY id DESC LIMIT {limit}")
        registros_raw = cur.fetchall()
        cur.close()
        registros = [dict(row) for row in registros_raw]
        print(f"ℹ️ Dados do Bot: Carregados {len(registros)} registros recentes da tabela 'grafica'.")
        return registros
    except psycopg2.errors.UndefinedTable:
         print(f"⚠️ AVISO: A tabela 'grafica' não foi encontrada. O chatbot não terá contexto de pedidos.")
         return []
    except Exception as e:
        print(f"❌ ERRO ao buscar dados da tabela 'grafica' para o bot: {e}")
        traceback.print_exc()
        return []
    finally:
        if conn: conn.close()

grafica_data_context = get_grafica_data_for_bot()
grafica_json_context = json.dumps(grafica_data_context, cls=DecimalEncoder, ensure_ascii=False, separators=(',', ':'))

# --- NOVO SYSTEM PROMPT (ALTERADO) ---
SYSTEM_PROMPT = f"""
# ALTERADO: Removido "Teclabel" e ajustada a persona
Você é o 'GrafiBot', um assistente virtual amigável e especialista, focado em ajudar orçamentistas de gráfica a obterem *estimativas* de orçamento para produtos gráficos.
Sua única fonte de verdade para estimativas é a base de dados de pedidos recentes em JSON fornecida abaixo.

--- BASE DE DADOS (Pedidos Recentes - JSON) ---
{grafica_json_context}
--- FIM DA BASE DE DADOS ---

**FLUXO DE CONVERSA PARA ORÇAMENTO (SIGA ESTRITAMENTE):**

1.  **Saudação Amigável e Apresentação:** Comece sempre com algo como: "Olá! 👋 Sou o GrafiBot, seu assistente virtual para orçamentistas de gráfica. Estou aqui para ajudar a estimar o valor do seu próximo pedido ou consultar registros recentes. Como posso te ajudar hoje?" # ALTERADO: Removido "Teclabel"
2.  **Identifique a Intenção (Orçamento):** Se o usuário expressar interesse em preço, orçamento, cotação ou valor:
    * **Pergunte o Essencial (1ª pergunta):** "Legal! Para começarmos, me diga qual **produto** você tem em mente e a **quantidade** aproximada."
    * **Colete Detalhes Essenciais (Perguntas seguintes, UMA DE CADA VEZ):** Baseado na resposta, pergunte educadamente pelos detalhes CHAVE que você vê na BASE DE DADOS (Material, Impressão, Tamanho). Exemplos:
        * "Entendido. E qual **material** você está pensando para essas etiquetas?"
        * "Perfeito. E como seria a **impressão**? (Ex: 4x0 cores, 1x0 cor, digital...)"
        * "Anotado! Qual o **tamanho** aproximado que você precisa (Largura x Altura em cm)?"
    * **Continue perguntando** até ter pelo menos: Produto, Quantidade, Material e Impressão. O tamanho é bom ter, mas opcional se não souber.
3.  **Confirme os Dados Coletados:** Antes de prosseguir, recapitule de forma clara: "Ok, vamos confirmar: Você precisa de [Quantidade] [Produto] em [Material], com impressão [Impressão] e tamanho aproximado [LxA cm, se informado]. É isso mesmo?"
4.  **Busque e Forneça a ESTIMATIVA (SEMPRE):** Se o usuário confirmar:
    * Procure na BASE DE DADOS por 1 ou 2 pedidos **o mais similares possível** (mesmo produto/material, quantidade próxima).
    * **APRESENTE A ESTIMATIVA:** "Com base em pedidos recentes parecidos que encontrei aqui, uma estimativa para o seu pedido seria **em torno de R$ XXX,XX**."
    * **JUSTIFIQUE COM EXEMPLO:** "Para você ter uma ideia, encontrei o pedido ID [ID do Exemplo], que foram [Qtd Exemplo] [Produto Exemplo] em [Material Exemplo], e o valor final ficou em R$ [Valor Exemplo]." (Use apenas UM exemplo claro).
    * **REFORCE QUE É ESTIMATIVA:** Conclua SEMPRE com: "**Lembre-se: este é apenas um valor estimado** baseado em pedidos anteriores, ok? Para um orçamento exato e formal, por favor, preencha o formulário de cadastro na página."
5.  **Se Não Achar Similar:** Seja honesto: "Hmm, não encontrei pedidos recentes muito parecidos com essas especificações na minha base para dar uma estimativa confiável 🤔. Recomendo preencher o formulário na página para receber um orçamento preciso da nossa equipe."

**OUTRAS REGRAS:**

* **Consulta de Vendas:** Se o usuário perguntar sobre vendas/pedidos recentes, liste os 3-5 exemplos mais recentes da BASE DE DADOS de forma resumida (ID, Produto, Qtd, Valor).
* **NÃO ALUCINE:** Jamais invente preços, produtos, materiais ou características. Se não está na base, não existe para você.
* **SEJA CONVERSACIONAL e PACIENTE:** Use emojis leves (👋, 👍, 🤔, ✅), seja educado e guie o usuário passo a passo.
* **FOCO NA GRÁFICA:** Responda apenas sobre orçamentos e pedidos da gráfica. Recuse educadamente outros assuntos.
"""
# --- FIM DO NOVO SYSTEM PROMPT ---


# Inicializa o Modelo e a Sessão de Chat
model = None
chat_session = None
try:
    if api_key:
        model = genai.GenerativeModel('gemini-flash-latest')
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": [SYSTEM_PROMPT]},
                # ALTERADO: Mensagem inicial do modelo
                {"role": "model", "parts": ["Olá! 👋 Sou o GrafiBot, seu assistente virtual para orçamentistas de gráfica. Estou aqui para ajudar a estimar o valor do seu próximo pedido ou consultar registros recentes. Como posso te ajudar hoje?"]}
            ]
        )
        print("✅ Modelo Gemini ('gemini-flash-latest') inicializado com o NOVO contexto.")
    else:
        print("⚠️ AVISO: API Key do Gemini não carregada. O chatbot não funcionará.")

except Exception as e:
    print(f"❌ ERRO CRÍTICO ao inicializar o GenerativeModel: {e}")
    traceback.print_exc()

# --- ROTAS DA APLICAÇÃO ---

@app.route('/')
def index():
    """Serve a página principal."""
    return render_template('index.html')

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

        print(f"💬 Mensagem do Usuário: {user_message}") # Log da mensagem recebida

        # Envia a mensagem para o Gemini
        response = chat_session.send_message(
            user_message,
            generation_config=genai.types.GenerationConfig(temperature=0.7), # Um pouco de criatividade
            safety_settings={'HATE': 'BLOCK_NONE', 'HARASSMENT': 'BLOCK_NONE',
                             'SEXUAL' : 'BLOCK_NONE', 'DANGEROUS' : 'BLOCK_NONE'}
        )

        # Log curto da resposta antes de enviar
        print(f"🤖 Resposta do Bot: {response.text[:100]}...")
        return jsonify({'reply': response.text})

    except genai.types.generation_types.StopCandidateException as stop_ex:
        print(f"⚠️ API BLOQUEOU a resposta por segurança: {stop_ex}")
        return jsonify({'reply': "Desculpe, não posso gerar uma resposta para essa solicitação específica. Posso ajudar com orçamentos ou consulta de pedidos?"})
    except Exception as e:
        print(f"❌ Erro ao chamar a API do Gemini: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Ocorreu um erro ao processar sua mensagem com a IA.'}), 503

# Rota para registrar o pedido na tabela 'grafica'
@app.route('/api/registrar_pedido', methods=['POST'])
def registrar_pedido():
    """Recebe dados do formulário HTML e insere na tabela 'grafica'."""
    dados = request.json
    conn = None
    print(f"ℹ️ Recebido POST em /api/registrar_pedido: {dados}")

    if not dados or 'quantidade' not in dados or 'produto' not in dados or 'valorFinal' not in dados:
         print("❌ Erro em /api/registrar_pedido: Dados incompletos recebidos.")
         return jsonify({'error': 'Dados incompletos para registrar o pedido.'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        sql_insert = """
        INSERT INTO grafica (
            QUANTIDADE, PRODUTO, MATERIAL, IMPRESSAO, LARGURA, ALTURA,
            TIPO_DE_CORTE, ACABAMENTO, EXTRA, VALOR_FINAL
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        valores = (
            dados.get('quantidade'), dados.get('produto'), dados.get('material'),
            dados.get('impressao'), dados.get('largura') or None, dados.get('altura') or None,
            dados.get('tipoCorte') or None, dados.get('acabamento'), dados.get('extra') or None,
            dados.get('valorFinal')
        )

        cur.execute(sql_insert, valores)
        conn.commit()
        cur.close()
        print("✅ Pedido registrado com sucesso na tabela 'grafica'.")

        # Recarrega o contexto do bot APÓS salvar o novo pedido
        global grafica_data_context, grafica_json_context, SYSTEM_PROMPT, chat_session
        print("🔄 Recarregando contexto do bot após novo pedido...")
        grafica_data_context = get_grafica_data_for_bot() # Busca os dados mais recentes
        grafica_json_context = json.dumps(grafica_data_context, cls=DecimalEncoder, ensure_ascii=False, separators=(',', ':'))

        # ATUALIZA O SYSTEM_PROMPT com os novos dados
        # (Precisa re-definir o SYSTEM_PROMPT aqui para incluir o novo JSON)
        SYSTEM_PROMPT = f"""
        # ALTERADO: Removido "Teclabel" e ajustada a persona
        Você é o 'GrafiBot', um assistente virtual amigável e especialista, focado em ajudar orçamentistas de gráfica a obterem *estimativas* de orçamento para produtos gráficos.
        Sua única fonte de verdade para estimativas é a base de dados de pedidos recentes em JSON fornecida abaixo.

        --- BASE DE DADOS (Pedidos Recentes - JSON) ---
        {grafica_json_context}
        --- FIM DA BASE DE DADOS ---

        **FLUXO DE CONVERSA PARA ORÇAMENTO (SIGA ESTRITAMENTE):**

        1.  **Saudação Amigável e Apresentação:** Comece sempre com algo como: "Olá! 👋 Sou o GrafiBot, seu assistente virtual para orçamentistas de gráfica. Estou aqui para ajudar a estimar o valor do seu próximo pedido ou consultar registros recentes. Como posso te ajudar hoje?" # ALTERADO: Removido "Teclabel"
        2.  **Identifique a Intenção (Orçamento):** Se o usuário expressar interesse em preço, orçamento, cotação ou valor:
            * **Pergunte o Essencial (1ª pergunta):** "Legal! Para começarmos, me diga qual **produto** você tem em mente e a **quantidade** aproximada."
            * **Colete Detalhes Essenciais (Perguntas seguintes, UMA DE CADA VEZ):** Baseado na resposta, pergunte educadamente pelos detalhes CHAVE que você vê na BASE DE DADOS (Material, Impressão, Tamanho). Exemplos:
                * "Entendido. E qual **material** você está pensando para essas etiquetas?"
                * "Perfeito. E como seria a **impressão**? (Ex: 4x0 cores, 1x0 cor, digital...)"
                * "Anotado! Qual o **tamanho** aproximado que você precisa (Largura x Altura em cm)?"
            * **Continue perguntando** até ter pelo menos: Produto, Quantidade, Material e Impressão. O tamanho é bom ter, mas opcional se não souber.
        3.  **Confirme os Dados Coletados:** Antes de prosseguir, recapitule de forma clara: "Ok, vamos confirmar: Você precisa de [Quantidade] [Produto] em [Material], com impressão [Impressão] e tamanho aproximado [LxA cm, se informado]. É isso mesmo?"
        4.  **Busque e Forneça a ESTIMATIVA (SEMPRE):** Se o usuário confirmar:
            * Procure na BASE DE DADOS por 1 ou 2 pedidos **o mais similares possível** (mesmo produto/material, quantidade próxima).
            * **APRESENTE A ESTIMATIVA:** "Com base em pedidos recentes parecidos que encontrei aqui, uma estimativa para o seu pedido seria **em torno de R$ XXX,XX**."
            * **JUSTIFIQUE COM EXEMPLO:** "Para você ter uma ideia, encontrei o pedido ID [ID do Exemplo], que foram [Qtd Exemplo] [Produto Exemplo] em [Material Exemplo], e o valor final ficou em R$ [Valor Exemplo]." (Use apenas UM exemplo claro).
            * **REFORCE QUE É ESTIMATIVA:** Conclua SEMPRE com: "**Lembre-se: este é apenas um valor estimado** baseado em pedidos anteriores, ok? Para um orçamento exato e formal, por favor, preencha o formulário de cadastro na página."
        5.  **Se Não Achar Similar:** Seja honesto: "Hmm, não encontrei pedidos recentes muito parecidos com essas especificações na minha base para dar uma estimativa confiável 🤔. Recomendo preencher o formulário na página para receber um orçamento preciso da nossa equipe."

        **OUTRAS REGRAS:**

        * **Consulta de Vendas:** Se o usuário perguntar sobre vendas/pedidos recentes, liste os 3-5 exemplos mais recentes da BASE DE DADOS de forma resumida (ID, Produto, Qtd, Valor).
        * **NÃO ALUCINE:** Jamais invente preços, produtos, materiais ou características. Se não está na base, não existe para você.
        * **SEJA CONVERSACIONAL e PACIENTE:** Use emojis leves (👋, 👍, 🤔, ✅), seja educado e guie o usuário passo a passo.
        * **FOCO NA GRÁFICA:** Responda apenas sobre orçamentos e pedidos da gráfica. Recuse educadamente outros assuntos.
        """

        # Reinicia a sessão de chat com o prompt atualizado
        if model:
             # Guarda o histórico antigo (opcional, pode ficar confuso)
             # old_history = chat_session.history if chat_session else []

             chat_session = model.start_chat(
                history=[
                    {"role": "user", "parts": [SYSTEM_PROMPT]},
                    # ALTERADO: Mensagem inicial do modelo após atualização
                    {"role": "model", "parts": ["Entendido. Sou o GrafiBot e minha base de pedidos foi atualizada com o último registro. Pronto para ajudar."]}
                    # Pode tentar adicionar o histórico antigo aqui se quiser manter a conversa:
                    # *old_history[2:] # Pula os prompts iniciais antigos
                ]
            )
             print("✅ Contexto do bot atualizado dinamicamente com o novo pedido.")
        else:
             print("⚠️ Bot não inicializado, não foi possível atualizar contexto.")

        return jsonify({'success': 'Pedido registrado! O chatbot já está ciente deste novo pedido.'}), 201

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
