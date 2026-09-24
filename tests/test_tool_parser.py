from core.tool_parser import StreamFilter, loads_lenient, parse_model_output

TOOLS = {"tavily_search", "run_terminal_command", "read_file"}


def test_hermes_json_call():
    out = parse_model_output('Vou pesquisar.\n<tool_call>\n{"name": "tavily_search", "arguments": {"query": "dólar hoje"}}\n</tool_call>', TOOLS)
    assert out.content == "Vou pesquisar."
    assert [(c.name, c.arguments) for c in out.tool_calls] == [("tavily_search", {"query": "dólar hoje"})]
    assert len(out.tool_calls[0].id) == 9 and out.tool_calls[0].id.isalnum()


def test_multiple_calls_and_string_arguments():
    text = ('<tool_call>{"name":"read_file","arguments":"{\\"path\\": \\"a.txt\\"}"}</tool_call>'
            '<tool_call>{"name":"run_terminal_command","parameters":{"command":"dir"}}</tool_call>')
    calls = parse_model_output(text, TOOLS).tool_calls
    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "a.txt"}),
                                                      ("run_terminal_command", {"command": "dir"})]


def test_qwen_xml_function_format():
    text = ("<tool_call>\n<function=run_terminal_command>\n<parameter=command>\npdftotext -layout \"a b.pdf\" -\n"
            "</parameter>\n<parameter=timeout_seconds>\n30\n</parameter>\n</function>\n</tool_call>")
    call = parse_model_output(text, TOOLS).tool_calls[0]
    assert call.name == "run_terminal_command"
    assert call.arguments == {"command": 'pdftotext -layout "a b.pdf" -', "timeout_seconds": 30}


def test_mistral_formats():
    a = parse_model_output('[TOOL_CALLS][{"name": "tavily_search", "arguments": {"query": "x"}}]', TOOLS)
    b = parse_model_output('[TOOL_CALLS]tavily_search[ARGS]{"query": "y"}', TOOLS)
    assert a.tool_calls[0].arguments == {"query": "x"} and a.content == ""
    assert (b.tool_calls[0].name, b.tool_calls[0].arguments) == ("tavily_search", {"query": "y"})


def test_gemma4_format():
    out = parse_model_output('<|tool_call>call:tavily_search{query:<|"|>preço, "ouro"<|"|>,max_results:3}<tool_call|>', TOOLS)
    assert out.tool_calls[0].arguments == {"query": 'preço, "ouro"', "max_results": 3}


def test_pythonic_lfm_format():
    out = parse_model_output('<|tool_call_start|>[read_file(path="c:/x.txt", offset=10)]<|tool_call_end|>', TOOLS)
    assert out.tool_calls[0].arguments == {"path": "c:/x.txt", "offset": 10}


def test_bare_json_only_for_known_tools():
    out = parse_model_output('```json\n{"name": "tavily_search", "arguments": {"query": "z"}}\n```', TOOLS)
    assert out.tool_calls and out.content == ""
    plain = parse_model_output('Exemplo: {"name": "foo", "arguments": {}}', TOOLS)
    assert plain.tool_calls == [] and "Exemplo" in plain.content


def test_thinking_split_and_prefilled_think():
    out = parse_model_output("<think>hmm\nok</think>\nResposta final.", TOOLS)
    assert out.thinking == "hmm\nok" and out.content == "Resposta final."
    pre = parse_model_output("pensando aqui</think>Resposta.", TOOLS, starts_in_thinking=True)
    assert pre.thinking == "pensando aqui" and pre.content == "Resposta."


def test_invalid_arguments_report_error():
    out = parse_model_output('<tool_call>{"name": "read_file", "arguments": {path: }}</tool_call>', TOOLS)
    assert out.tool_calls[0].error


def test_lenient_json():
    assert loads_lenient("{'a': True, 'b': None,}") == {"a": True, "b": None}
    assert loads_lenient('texto {"a": {"b": 1}} fim') == {"a": {"b": 1}}


def test_stream_filter_hides_tool_calls_and_splits_thinking():
    f = StreamFilter()
    events = []
    for piece in ["Olá <th", "ink>raciocínio</thi", "nk>visível <tool", "_call>{\"name\":1}</tool_call> fim"]:
        events += f.feed(piece)
    events += f.flush()
    text = "".join(t for k, t in events if k == "text")
    think = "".join(t for k, t in events if k == "think")
    assert text == "Olá visível  fim"
    assert think == "raciocínio"
    assert any(k == "tool_start" for k, _ in events)
    assert not any("name" in t for k, t in events if k == "text")


def test_stream_filter_starts_in_thinking():
    f = StreamFilter(starts_in_thinking=True)
    events = f.feed("pensamento</think>resposta") + f.flush()
    assert ("think", "pensamento") in events and ("text", "resposta") in events
