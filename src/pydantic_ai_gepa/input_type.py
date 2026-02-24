"""Structured input utilities for GEPA optimization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import html
import json
from dataclasses import is_dataclass, replace
from typing import Annotated, Any, Generic, TypeVar, get_args, get_origin

try:
    import pydantic_monty
except ImportError:
    pydantic_monty = None

from pydantic import BaseModel

from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    UserContent,
    VideoUrl,
)

from pydantic_ai_gepa.gepa_graph.models.candidate import CandidateMap

__all__ = [
    "SignatureSuffix",
    "generate_system_instructions",
    "generate_user_content",
    "get_gepa_components",
    "apply_candidate_to_input_model",
    "extract_signature_components",
    "BoundInputSpec",
    "InputSpec",
    "build_input_spec",
]

AttachmentContent = AudioUrl | BinaryContent | DocumentUrl | ImageUrl | VideoUrl
ATTACHMENT_TYPE_TO_LABEL: dict[type, str] = {
    ImageUrl: "image",
    VideoUrl: "video",
    AudioUrl: "audio",
    DocumentUrl: "document",
    BinaryContent: "binary",
}

_DEFAULT_ENCODER_SCRIPT = """
def encode_input(data):
    return "<input>\\n" + json_dumps(data) + "\\n</input>"

encode_input(data)
"""


class SignatureSuffix:
    """Marker for fields that should be appended as plain text without formatting."""


class _AttachmentRegistry:
    """Track multimodal attachments and provide textual references."""

    def __init__(self) -> None:
        self.attachments: list[AttachmentContent] = []
        self.placeholders: list[str] = []
        self._placeholders: dict[int, str] = {}

    def register(self, content: AttachmentContent) -> str:
        key = id(content)
        existing = self._placeholders.get(key)
        if existing is not None:
            return existing

        label = self._label_for(content)
        # Use the content's stable identifier (hash-based) for consistent references
        ref = content.identifier
        placeholder = f'<{label} ref="{ref}"/>'
        self.attachments.append(content)
        self.placeholders.append(placeholder)
        self._placeholders[key] = placeholder
        return placeholder

    @staticmethod
    def _label_for(content: AttachmentContent) -> str:
        match content:
            case ImageUrl():
                return "image"
            case VideoUrl():
                return "video"
            case AudioUrl():
                return "audio"
            case DocumentUrl():
                return "document"
            case _:
                return "binary"


def generate_system_instructions(
    instance: BaseModel,
    *,
    candidate: dict[str, str] | None = None,
) -> str:
    """Generate system instructions for a structured input instance."""
    view = _InputModelView(instance)
    return view.build_system_instructions(candidate=candidate)


def generate_user_content(
    instance: BaseModel,
    *,
    candidate: dict[str, str] | None = None,
) -> Sequence[UserContent]:
    """Convert a structured input instance to user content."""
    view = _InputModelView(instance)
    return view.build_user_content(candidate=candidate)


def get_gepa_components(model_cls: type[BaseModel]) -> dict[str, str]:
    """Extract default GEPA components for a structured input model."""
    class_view = _InputClassView(model_cls)
    return class_view.get_gepa_components()


@contextmanager
def apply_candidate_to_input_model(
    model_cls: type[BaseModel],
    candidate: CandidateMap | None,
) -> Iterator[None]:
    """Temporarily apply a GEPA candidate to a structured input model."""
    class_view = _InputClassView(model_cls)
    with class_view.apply_candidate(candidate):
        yield


def extract_signature_components(
    models: Sequence[type[BaseModel]],
) -> dict[str, str]:
    """Extract all GEPA components from multiple structured input models."""
    all_components: dict[str, str] = {}
    for model_cls in models:
        all_components.update(get_gepa_components(model_cls))
    return all_components


ModelT = TypeVar("ModelT", bound=BaseModel)


class BoundInputSpec(Generic[ModelT]):
    """Normalized view over a structured input model."""

    def __init__(self, model_cls: type[ModelT]) -> None:
        if not issubclass(model_cls, BaseModel):
            raise TypeError(
                f"Input specs must be Pydantic BaseModel subclasses, got {model_cls!r}"
            )
        self.model_cls: type[ModelT] = model_cls

    def generate_system_instructions(
        self,
        instance: ModelT,
        *,
        candidate: dict[str, str] | None = None,
    ) -> str:
        return generate_system_instructions(instance, candidate=candidate)

    def generate_user_content(
        self,
        instance: ModelT,
        *,
        candidate: dict[str, str] | None = None,
    ) -> Sequence[UserContent]:
        return generate_user_content(instance, candidate=candidate)

    def get_gepa_components(self) -> dict[str, str]:
        return get_gepa_components(self.model_cls)

    @contextmanager
    def apply_candidate(
        self,
        candidate: CandidateMap | None,
    ) -> Iterator[None]:
        with apply_candidate_to_input_model(self.model_cls, candidate):
            yield


InputSpec = type[ModelT] | BoundInputSpec[ModelT]


def build_input_spec(input_spec: InputSpec[ModelT]) -> BoundInputSpec[ModelT]:
    """Normalize an input specification into a BoundInputSpec."""
    if isinstance(input_spec, BoundInputSpec):
        return input_spec
    return BoundInputSpec(input_spec)


class _InputShared:
    """Utilities shared between instance-level and class-level operations."""

    def __init__(self, model_cls: type[BaseModel]) -> None:
        self.model_cls = model_cls

    @staticmethod
    def _unwrap_annotated(annotation: Any) -> Any:
        """Strip Annotated metadata to expose the underlying type."""
        current = annotation
        while True:
            origin = get_origin(current)
            if origin is Annotated:
                args = get_args(current)
                if not args:
                    break
                current = args[0]
                continue
            break
        return current

    def _get_effective_text(
        self,
        component_key: str,
        default: str,
        candidate: dict[str, str] | None,
    ) -> str:
        if candidate is None:
            return default
        full_key = f"signature:{self.model_cls.__name__}:{component_key}"
        return candidate.get(full_key, default)

    @staticmethod
    def _is_suffix_field(field_info: Any) -> bool:
        if getattr(field_info, "metadata", None):
            return SignatureSuffix in field_info.metadata or any(
                isinstance(m, type) and issubclass(m, SignatureSuffix)
                for m in field_info.metadata
            )
        return False

    @staticmethod
    def _get_type_name(annotation: Any) -> str:
        annotation = _InputShared._unwrap_annotated(annotation)
        if annotation is None:
            return "Any"

        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is None:
            if isinstance(annotation, type):
                if issubclass(annotation, BaseModel):
                    return annotation.__name__
                return annotation.__name__
            return str(annotation)

        origin_name = getattr(origin, "__name__", str(origin))
        if not args:
            return origin_name

        arg_names = ", ".join(
            _InputShared._get_type_name(arg)
            for arg in args  # type: ignore[attr-defined]
        )
        return f"{origin_name}[{arg_names}]"

    @staticmethod
    def _collect_attachment_labels(annotation: Any) -> set[str]:
        labels: set[str] = set()
        annotation = _InputShared._unwrap_annotated(annotation)

        origin = get_origin(annotation)
        if origin is None:
            if isinstance(annotation, type):
                label = ATTACHMENT_TYPE_TO_LABEL.get(annotation)
                if label:
                    labels.add(label)
        else:
            args = get_args(annotation)
            if not args:
                return labels
            if origin in (list, tuple, set, frozenset, Sequence):
                labels.update(_InputShared._collect_attachment_labels(args[0]))
            else:
                for arg in args:
                    labels.update(_InputShared._collect_attachment_labels(arg))

        return labels

    @staticmethod
    def _attachment_note_for_annotation(annotation: Any) -> str | None:
        labels = _InputShared._collect_attachment_labels(annotation)
        if not labels:
            return None
        if len(labels) == 1:
            label = next(iter(labels))
            article = "an" if label[0].lower() in {"a", "e", "i", "o", "u"} else "a"
            return f"Provide {article} {label} reference using the appropriate attachment type."
        joined = ", ".join(sorted(labels))
        return f"Provide references using the appropriate attachment types ({joined})."

    @staticmethod
    def _format_field_label(field_name: str) -> str:
        parts = field_name.replace("_", " ").strip().split()
        if not parts:
            return field_name
        return " ".join(part.capitalize() for part in parts)

    @staticmethod
    def _is_simple_scalar(value: Any) -> bool:
        return isinstance(value, (str, int, float, bool)) or value is None

    @staticmethod
    def _is_list_of_models(value: Any) -> bool:
        return (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(item, BaseModel) for item in value)
        )

    @staticmethod
    def _get_model_type_from_annotation(field_type: Any) -> type[BaseModel] | None:
        field_type = _InputShared._unwrap_annotated(field_type)
        if field_type is None:
            return None
        origin = get_origin(field_type)
        if origin is not None:
            args = get_args(field_type)
            if args:
                inner_type = args[0]
                if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                    return inner_type
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return field_type
        return None

    def _extract_model_schema(self, model_class: type[BaseModel]) -> dict[str, str]:
        schema: dict[str, str] = {}
        for field_name, field_info in model_class.model_fields.items():
            description = field_info.description or f"The {field_name} field"
            note = self._attachment_note_for_annotation(field_info.annotation)
            if note:
                description = f"{description}. {note}" if description else note
            schema[field_name] = description
        return schema

    def _format_fields_recursive(
        self,
        model_class: type[BaseModel],
        indent: int = 0,
        visited: set[type[BaseModel]] | None = None,
    ) -> list[str]:
        if visited is None:
            visited = set()
        if model_class in visited:
            return []
        visited.add(model_class)

        schema = self._extract_model_schema(model_class)
        lines: list[str] = []
        indent_str = " " * indent

        lines.append(f"{indent_str}{model_class.__name__}")
        for field_name, description in schema.items():
            field = model_class.model_fields[field_name]
            field_annotation = field.annotation
            type_name = self._get_type_name(field_annotation)
            desc = description or f"The {field_name} field"
            lines.append(f"{indent_str}  - `<{field_name}>` ({type_name}): {desc}")

            nested_model = self._get_model_type_from_annotation(field_annotation)
            if nested_model:
                nested_lines = self._format_fields_recursive(
                    nested_model, indent + 4, visited
                )
                lines.extend(nested_lines)

        return lines

    def _format_model_schema(self, model_class: type[BaseModel]) -> str | None:
        lines = self._format_fields_recursive(model_class)
        if not lines:
            return None
        return "\n".join(lines)


class _InputModelView(_InputShared):
    """Operations that require a concrete model instance."""

    def __init__(self, instance: BaseModel) -> None:
        super().__init__(instance.__class__)
        self.instance = instance

    def build_system_instructions(
        self,
        *,
        candidate: dict[str, str] | None,
    ) -> str:
        instructions = self._get_effective_text(
            "instructions", self.model_cls.__doc__ or "", candidate
        )

        instruction_sections: list[str] = []
        suffix_parts: list[str] = []
        input_lines: list[str] = []
        schema_descriptions: dict[type[BaseModel], str] = {}

        if instructions:
            instruction_sections.append(instructions.strip())

        for field_name, field_info in self.model_cls.model_fields.items():
            if self._is_suffix_field(field_info):
                default_suffix = (
                    field_info.default if field_info.default is not None else ""
                )
                suffix_text = self._get_effective_text(
                    field_name, str(default_suffix), candidate
                )
                if suffix_text:
                    suffix_parts.append(suffix_text)
                continue

            default_desc = field_info.description or f"The {field_name} input"
            field_desc = self._get_effective_text(
                f"{field_name}:desc", default_desc, candidate
            )
            note = self._attachment_note_for_annotation(field_info.annotation)
            if note:
                field_desc = f"{field_desc}. {note}" if field_desc else note

            if field_desc:
                type_name = self._get_type_name(field_info.annotation)
                input_lines.append(f"- `<{field_name}>` ({type_name}): {field_desc}")

            model_type = self._get_model_type_from_annotation(field_info.annotation)
            if model_type and model_type not in schema_descriptions:
                schema_desc = self._format_model_schema(model_type)
                if schema_desc:
                    schema_descriptions[model_type] = schema_desc

        if input_lines:
            instruction_sections.append("\nInputs:")
            instruction_sections.append("\n".join(input_lines))

        if schema_descriptions:
            instruction_sections.append("\nSchemas:")
            instruction_sections.append("\n\n".join(schema_descriptions.values()))

        if suffix_parts:
            suffix_text = "\n".join(part.strip() for part in suffix_parts if part)
            if suffix_text:
                instruction_sections.append(suffix_text)

        result = "\n\n".join(
            section.strip() for section in instruction_sections if section.strip()
        )
        # Prepend a newline so that when pydantic-ai joins multiple instruction parts
        # with '\n', we get proper '\n\n' separation between base and signature instructions
        return f"\n{result}" if result else result

    def build_user_content(
        self,
        *,
        candidate: dict[str, str] | None = None,
    ) -> Sequence[UserContent]:
        encoder_key = f"signature:{self.model_cls.__name__}:encoder"
        encoder_script = candidate.get(encoder_key) if candidate else None
        
        registry = _AttachmentRegistry()
        transformed_instance = self._replace_attachments_with_refs(self.instance, registry)

        if encoder_script and pydantic_monty is not None:
            try:
                data_dict = transformed_instance.model_dump(mode="json")
                
                def host_json_dumps(obj: Any) -> str:
                    return json.dumps(obj, indent=2)

                m = pydantic_monty.Monty(
                    encoder_script,
                    inputs=["data"],
                    external_functions=["json_dumps"]
                )
                encoded_str = m.run(
                    inputs={"data": data_dict},
                    external_functions={"json_dumps": host_json_dumps}
                )

                user_content: list[UserContent] = []
                user_content.extend(registry.attachments)

                if isinstance(encoded_str, str) and encoded_str:
                    full_prompt = encoded_str
                    for placeholder in registry.placeholders:
                        escaped = html.escape(placeholder, quote=False)
                        full_prompt = full_prompt.replace(escaped, placeholder)
                    user_content.append(full_prompt)

                return user_content
            except Exception as e:
                import logfire
                logfire.warning("Failed to execute monty encoder script, falling back to XML", exc_info=e)

        content_sections: list[str] = []

        for field_name, field_info in self.model_cls.model_fields.items():
            field_value = getattr(transformed_instance, field_name)

            if field_value is None:
                continue

            if self._is_suffix_field(field_info):
                continue

            formatted_value = self._format_field_value_xml(
                field_name, field_value
            )
            if formatted_value:
                content_sections.append(formatted_value)

        full_prompt = "\n\n".join(
            section.strip() for section in content_sections if section.strip()
        )
        user_content: list[UserContent] = []
        user_content.extend(registry.attachments)
        if full_prompt:
            for placeholder in registry.placeholders:
                escaped = html.escape(placeholder, quote=False)
                full_prompt = full_prompt.replace(escaped, placeholder)
            user_content.append(full_prompt)
        return user_content

    def _replace_attachments_with_refs(
        self,
        value: Any,
        registry: _AttachmentRegistry,
    ) -> Any:
        if isinstance(
            value, (ImageUrl, VideoUrl, AudioUrl, DocumentUrl, BinaryContent)
        ):
            return registry.register(value)

        if isinstance(value, str) or value is None:
            return value

        if isinstance(value, (int, float, bool)):
            return value

        if isinstance(value, BaseModel):
            updated = {
                key: self._replace_attachments_with_refs(getattr(value, key), registry)
                for key in value.__class__.model_fields
            }
            return value.model_copy(update=updated)

        if is_dataclass(value) and not isinstance(value, type):
            updated = {
                field_name: self._replace_attachments_with_refs(
                    getattr(value, field_name), registry
                )
                for field_name in value.__dataclass_fields__
            }
            return replace(value, **updated)

        if isinstance(value, Mapping):
            return {
                key: self._replace_attachments_with_refs(val, registry)
                for key, val in value.items()
            }

        if isinstance(value, Sequence) and not isinstance(
            value, (bytes, bytearray, str)
        ):
            return [
                self._replace_attachments_with_refs(item, registry) for item in value
            ]

        return value

    def _format_field_value_xml(
        self,
        field_name: str,
        field_value: Any,
    ) -> str | None:
        if field_value is None:
            return None

        if self._is_list_of_models(field_value):
            if not field_value:
                return f"<{field_name}></{field_name}>"

            item_class = field_value[0].__class__
            return format_as_xml(
                field_value,
                root_tag=field_name,
                item_tag=item_class.__name__,
                indent="  ",
            )

        return format_as_xml(
            field_value,
            root_tag=field_name,
            item_tag="item",
            indent="  ",
        )


class _InputClassView(_InputShared):
    """Operations that only need the model class."""

    def get_gepa_components(self) -> dict[str, str]:
        components: dict[str, str] = {}

        if self.model_cls.__doc__:
            components[f"signature:{self.model_cls.__name__}:instructions"] = (
                self.model_cls.__doc__ or ""
            )

        for field_name, field_info in self.model_cls.model_fields.items():
            desc = field_info.description or f"The {field_name} input"
            components[f"signature:{self.model_cls.__name__}:{field_name}:desc"] = desc

        components[f"signature:{self.model_cls.__name__}:encoder"] = _DEFAULT_ENCODER_SCRIPT

        return components

    @contextmanager
    def apply_candidate(
        self,
        candidate: CandidateMap | None,
    ) -> Iterator[None]:
        if candidate is None:
            yield
            return

        original_instructions = self.model_cls.__doc__
        original_descs: dict[str, str] = {}

        for field_name, field_info in self.model_cls.model_fields.items():
            original_descs[field_name] = field_info.description or ""

        try:
            instructions_key = f"signature:{self.model_cls.__name__}:instructions"
            if instructions_key in candidate:
                self.model_cls.__doc__ = candidate[instructions_key].text

            for field_name, field_info in self.model_cls.model_fields.items():
                desc_key = f"signature:{self.model_cls.__name__}:{field_name}:desc"
                if desc_key in candidate:
                    field_info.description = candidate[desc_key].text
            yield
        finally:
            self.model_cls.__doc__ = original_instructions
            for field_name, field_info in self.model_cls.model_fields.items():
                field_info.description = original_descs.get(field_name, "")
