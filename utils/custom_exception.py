import sys
from typing import Optional

class CustomException(Exception):
    def __init__(self, message: str, error_detail: Optional[Exception] = None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(message: str, error_detail: Optional[Exception]) -> str:
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown File"
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown Line"
        detail_str = str(error_detail) if error_detail else "None"
        return f"{message} | Error Detail: {detail_str} | File: {file_name} | Line: {line_number}"

    def __str__(self) -> str:
        return self.error_message
