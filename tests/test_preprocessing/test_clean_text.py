import pytest

from mini_rag.preprocessing import clean_text


@pytest.mark.parametrize(
    "input,output",
    [
        (
            "This method is used frequently [1]. It is based on earlier studies [2, 3].",
            "This method is used frequently. It is based on earlier studies.",
        ),
        (
            "This method is used frequently [1,7]. It is based on earlier studies [2,3].",
            "This method is used frequently. It is based on earlier studies.",
        ),
        (
            "This has been shown previously (Smith et al., 2021).",
            "This has been shown previously.",
        ),
        (
            "This has been shown previously in year 2000 (Smith et al., 2021).",
            "This has been shown previously in year 2000.",
        ),
        ("This    is   a test.  ", "This is a test."),
        (
            "This is a sentence . And another one , too !",
            "This is a sentence. And another one, too!",
        ),
        (
            "Studies [2] show (Doe, 2020) results are valid .",
            "Studies show results are valid.",
        ),
        ("", ""),
        ("[1] (Smith et al., 2020)", ""),
        (
            "No citations here. This is just plain text.",
            "No citations here. This is just plain text.",
        ),
    ],
)
def test_clean_text(input, output):
    """Tests the text cleaning function."""
    cleaned = clean_text(text=input)
    assert cleaned == output
