class AnswerStreamParser:

    def __init__(self):
        self.buffer = ""
        self.answer_started = False
        self.answer_finished = False
        self.scan_position = 0

    def feed(self, delta: str) -> str:
        """
        Feed one streamed JSON chunk.

        Returns only the newly available text from the
        top-level `answer` field.
        """

        self.buffer += delta

        if self.answer_finished:
            return ""

        # Find the beginning of the answer field.
        if not self.answer_started:
            marker = '"answer":"'
            start = self.buffer.find(marker)

            if start == -1:
                return ""

            self.answer_started = True
            self.scan_position = start + len(marker)

        output = []

        i = self.scan_position

        while i < len(self.buffer):

            char = self.buffer[i]

            # -------------------------------------------------
            # Handle escaped characters
            # -------------------------------------------------

            if char == "\\":
                # Escape sequence is incomplete.
                if i + 1 >= len(self.buffer):
                    break

                next_char = self.buffer[i + 1]

                escape_map = {
                    '"': '"',
                    "\\": "\\",
                    "/": "/",
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    "b": "\b",
                    "f": "\f",
                }

                # Handle unicode escape separately.
                if next_char == "u":

                    # Need \uXXXX completely available.
                    if i + 5 >= len(self.buffer):
                        break

                    hex_value = self.buffer[i + 2:i + 6]

                    try:
                        output.append(chr(int(hex_value, 16)))
                    except ValueError:
                        raise ValueError(
                            f"Invalid unicode escape: {hex_value}"
                        )

                    i += 6
                    continue

                output.append(
                    escape_map.get(next_char, next_char)
                )

                i += 2
                continue

            # -------------------------------------------------
            # Closing quote of answer
            # -------------------------------------------------

            if char == '"':
                self.answer_finished = True
                self.scan_position = i + 1
                break

            output.append(char)
            i += 1

        self.scan_position = i

        return "".join(output)