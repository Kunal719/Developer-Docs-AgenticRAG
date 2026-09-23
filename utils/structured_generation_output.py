def make_openai_strict(schema: dict) -> None:
    if schema.get("type") == "object":
        properties = schema.get("properties", {})

        schema["additionalProperties"] = False
        schema["required"] = list(properties.keys())

    for definition in schema.get("$defs", {}).values():
        make_openai_strict(definition)