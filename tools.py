from fpdf import FPDF
from browser_use import Tools, ActionResult, BrowserSession
import PyPDF2
import os

tools = Tools()

@tools.action(description='Cria um arquivo PDF com o conteúdo fornecido. Requer o nome do arquivo e o conteúdo.')
def create_pdf_file(file_name: str, content: str) -> ActionResult:
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', size=12)
        # Certifique-se de que o conteúdo é uma string e lida com caracteres especiais
        pdf.multi_cell(0, 10, content.encode('latin-1', 'replace').decode('latin-1'))
        pdf.output(file_name)
        return ActionResult(extracted_content=f'PDF {file_name} criado com sucesso.')
    except Exception as e:
        return ActionResult(extracted_content=f'Erro ao criar PDF {file_name}: {e}')

@tools.action(description='Realiza uma operação matemática e retorna o resultado. Requer a expressão matemática como string.')
def calculate(expression: str) -> ActionResult:
    try:
        result = eval(expression) # Cuidado com eval em produção, mas para uso local é aceitável
        return ActionResult(extracted_content=f'O resultado de {expression} é {result}.')
    except Exception as e:
        return ActionResult(extracted_content=f'Erro ao calcular {expression}: {e}')

@tools.action(description='Lê o conteúdo de um arquivo PDF e retorna o texto. Requer o caminho completo do arquivo PDF.')
def read_pdf_content(file_path: str) -> ActionResult:
    try:
        if not os.path.exists(file_path):
            return ActionResult(extracted_content=f'Erro: Arquivo PDF não encontrado em {file_path}')
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or ''
            return ActionResult(extracted_content=f'Conteúdo do PDF {file_path}:\n{text}')
    except Exception as e:
        return ActionResult(extracted_content=f'Erro ao ler PDF {file_path}: {e}')

# Exemplo de ferramenta para navegar e extrair informações (usando browser_session)
@tools.action(description='Navega para uma URL e extrai o título da página. Requer a URL.')
async def navigate_and_get_title(browser_session: BrowserSession, url: str) -> ActionResult:
    try:
        page = await browser_session.must_get_current_page()
        await page.goto(url)
        title = await page.title()
        return ActionResult(extracted_content=f'Navegou para {url}. Título da página: {title}')
    except Exception as e:
        return ActionResult(extracted_content=f'Erro ao navegar ou obter título de {url}: {e}')
