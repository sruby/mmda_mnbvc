import xml.etree.ElementTree as ET

# Define the namespaces used in the XML document
namespaces = {
    'tei': 'http://www.tei-c.org/ns/1.0',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
    'xlink': 'http://www.w3.org/1999/xlink'
    # Add other namespaces if needed
}


def parse_formula_to_latex(formula_element):
    # Extract the text content of the formula element
    formula_text = formula_element.text.strip()
    # Replace the formula with LaTeX delimiters for display math mode
    latex_formula = f"$$\n{formula_text}\n$$"
    return latex_formula


def parse_table_to_markdown(table_element):

    # 开始Markdown表格
    md_table = ""

    # 遍历每一行
    for row in table_element.findall('.//row'):
        md_row = []
        for cell in row.findall('.//cell'):
            # 检查是否有跨列的属性
            colspan = cell.get('cols')
            cell_text = cell.text.strip() if cell.text else ""
            if colspan:
                # 对于跨列的单元格，我们将重复该单元格内容
                md_row.extend([cell_text] * int(colspan))
            else:
                md_row.append(cell_text)
        # 创建Markdown表格的行
        md_table += "| " + " | ".join(md_row) + " |\n"

    # 创建Markdown表格的分隔符行
    # 假设所有列的格式都是相同的
    headers = md_table.split('\n')[0].split('|')[1:-1]  # 第一个和最后一个元素是空字符串，所以跳过
    separator = "| " + " | ".join(['---'] * len(headers)) + " |\n"

    # 将分隔符行插入表头下方
    md_table = md_table.split('\n')
    md_table.insert(1, separator)

    return '\n'.join(md_table) + "\n"


def convert(xml_content):
    # Parse the XML content
    root = ET.fromstring(xml_content)

    # Extract and format the title
    title = root.find('.//tei:titleStmt/tei:title', namespaces=namespaces).text
    mdContent = f'# {title}\n'

    # Extract and format the authors' names
    authors = root.findall('./tei:teiHeader//tei:author/tei:persName', namespaces=namespaces)
    for author in authors:
        forename = author.find('tei:forename', namespaces=namespaces)
        if forename is not None:
            forename = forename.text
        surname = author.find('tei:surname', namespaces=namespaces)

        if surname is not None:
            surname = surname.text
        mdContent += f'- {forename} {surname}\n'
    mdContent += '\n'

    # Extract and format the abstract
    abstract = root.find('.//tei:abstract', namespaces=namespaces)
    mdContent += f'## Abstract\n'
    abstract_s_list = abstract.findall('.//tei:s', namespaces=namespaces)
    for abstract_s in abstract_s_list:
        mdContent += f'{abstract_s.text}'

    # extract and format the body
    body_div_list = root.findall('./tei:text/tei:body/tei:div', namespaces=namespaces)
    for body_div in body_div_list:
        div_head = body_div.find('./tei:head', namespaces=namespaces)
        head_n = div_head.get('n')
        head_text = div_head.text
        if head_n is not None:
            head_text = head_n + ' '+ head_text
        mdContent += f'\n## {head_text}\n'

        tag_prefix =  '{http://www.tei-c.org/ns/1.0}'
        for child  in body_div:
            print("child_tag:"+child.tag , child.attrib)
            if child.tag == tag_prefix + 'p':
                # 段落前增加空格
                mdContent +='  '
                body_div_p_s_list = child.findall('tei:s', namespaces=namespaces)
                for body_div_p_s in body_div_p_s_list :
                    body_div_p_s_text = body_div_p_s.text
                    print("body_div_p_s_text:"+body_div_p_s_text)
                    mdContent += f'{body_div_p_s_text}'
                #     parse ref
                    # Iterate over the XML elements inside the s node
                    for elem in body_div_p_s:
                        # Check for 'ref' tag and append appropriate markdown
                        if elem.tag == tag_prefix + 'ref':
                            ref_type = elem.attrib.get('type')
                            if ref_type == 'bibr':
                                # Convert bibliography reference to markdown citation format
                                mdContent+=(f"[{elem.text}]")
                                print('markdown_elements_bibr:', mdContent)
                            elif ref_type == 'figure':
                                # Convert figure reference to markdown image format (as a placeholder here)
                                mdContent+=(f"{elem.text}")
                                ref_string = ET.tostring(elem, encoding='unicode')
                                mdContent += (f"{ref_string}")
                                print('markdown_elements_figure:', mdContent)
                        # Append the tail text if it exists (text after a subelement)
                        if elem.tail:
                            print(elem.tail)
                            mdContent+=(elem.tail)
                mdContent += '\n'
            elif child.tag == tag_prefix +'formula':
                mdContent += ET.tostring(child, encoding='unicode')
                mdContent += parse_formula_to_latex(child)
            else:
                print('warning:'+child.tag)

    # extract figure
    # figureDict = {}
    # body_figure_list = root.findall('./tei:text/tei:body/tei:figure', namespaces=namespaces)
    # for body_figure in body_figure_list:
    #     figure_type = body_figure.get('type')
    #     figureid = body_figure.get('xml:id')
    #     if figure_type is not None and figure_type == 'table':
    #         tablecontent = parse_table_to_markdown(body_figure)
    #         figureDict.put(figureid, tablecontent)
    #     else:
    #         figure_s = body_figure.findall('.//tei:s', namespaces=namespaces)
    #         figure_graphic = ''
    #         for figure_s in figure_s:
    #             figure_graphic += f'{figure_s.text}'


    # extract reference


    return mdContent



# parse
XML_PATH = './e5910c027af0ee9c1901c57f6579d903aedee7f4.xml'
xml = open(XML_PATH, encoding='utf-8').read()
mdContent = convert(xml)
with open('output.md', 'w', encoding='utf-8') as f:
        f.write(mdContent)
