from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field
from pydantic_ai.messages import ImageUrl, UserContent

from pydantic_ai_gepa.input_type import (
    generate_system_instructions,
    generate_user_content,
)

from inline_snapshot import snapshot


class GallerySignature(BaseModel):
    """Structured input mixing multimodal content with text."""

    gallery: Sequence[UserContent] = Field(description="Gallery content to inspect")
    notes: str = Field(description="Natural-language instructions")


def test_user_content_with_multimodal_resources() -> None:
    first = ImageUrl(url="https://example.com/a.png")
    second = ImageUrl(url="https://example.com/b.png")

    sig = GallerySignature(
        gallery=[
            first,
            "Provide a comparison of the screenshots.",
            second,
            "Highlight any mismatched UI states.",
        ],
        notes="Focus your answer on layout differences.",
    )

    user_content = generate_user_content(sig)
    images = user_content[:2]
    assert all([isinstance(image, ImageUrl) for image in images])
    text_content = user_content[2]
    assert text_content == snapshot("""\
<gallery>
  <item><image ref="b86daf"/></item>
  <item>Provide a comparison of the screenshots.</item>
  <item><image ref="a98844"/></item>
  <item>Highlight any mismatched UI states.</item>
</gallery>

<notes>Focus your answer on layout differences.</notes>\
""")


class ReferenceModel(BaseModel):
    attachment: ImageUrl
    remark: str


class NestedSignature(BaseModel):
    reference: ReferenceModel = Field(description="Structured reference data")
    repeated: Sequence[UserContent] = Field(
        description="Repeated attachments for reuse"
    )


def test_duplicate_attachment_reuses_placeholder() -> None:
    screenshot = ImageUrl(url="https://example.com/reused.png")

    sig = NestedSignature(
        reference=ReferenceModel(attachment=screenshot, remark="Primary capture."),
        repeated=["Revisit the earlier capture here:", screenshot],
    )

    system_instructions = generate_system_instructions(sig)
    assert system_instructions == snapshot("""
Inputs:

- `<reference>` (ReferenceModel): Structured reference data
- `<repeated>` (Sequence[Union[str, UnionType[ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent], CachePoint]]): Repeated attachments for reuse. Provide references using the appropriate attachment types (audio, binary, document, image, video).

Schemas:

ReferenceModel
  - `<attachment>` (ImageUrl): The attachment field. Provide an image reference using the appropriate attachment type.
  - `<remark>` (str): The remark field\
""")

    user_content = generate_user_content(sig)
    image = user_content[0]
    assert isinstance(image, ImageUrl)
    text_content = user_content[1]
    assert text_content == snapshot("""\
<reference>
  <attachment><image ref="cd6702"/></attachment>
  <remark>Primary capture.</remark>
</reference>

<repeated>
  <item>Revisit the earlier capture here:</item>
  <item><image ref="cd6702"/></item>
</repeated>\
""")
