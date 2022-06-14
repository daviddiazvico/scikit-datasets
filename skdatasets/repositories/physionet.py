from __future__ import annotations

import ast
import math
import re
import urllib
from html.parser import HTMLParser
from pathlib import Path
from typing import (
    Any,
    Final,
    List,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    overload,
)

import numpy as np
import pandas as pd
import wfdb.io
from sklearn.utils import Bunch

from .base import DatasetNotFoundError, fetch_zip

BASE_URL: Final = "https://physionet.org/static/published-projects"
INFO_STRING_SEMICOLONS_ONE_STR: Final = "(\S*): (\S*)"
INFO_STRING_SEMICOLONS_SEVERAL_REGEX: Final = re.compile(
    f"({INFO_STRING_SEMICOLONS_ONE_STR})+",
)
INFO_STRING_SEMICOLONS_ONE_REGEX: Final = re.compile(
    INFO_STRING_SEMICOLONS_ONE_STR,
)


class _ZipNameHTMLParser(HTMLParser):
    """Class for parsing the zip name in PhysioNet directory listing."""

    def __init__(self, *, convert_charrefs: bool = True) -> None:
        super().__init__(convert_charrefs=convert_charrefs)

        self.zip_name: str | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: List[Tuple[str, str | None]],
    ) -> None:
        if tag == "a":
            for attr in attrs:
                if attr[0] == "href" and attr[1] and attr[1].endswith(".zip"):
                    self.zip_name = attr[1]


def _get_zip_name_online(dataset_name: str) -> str:
    """Get the zip name of the dataset."""
    parser = _ZipNameHTMLParser()

    url_request = urllib.request.Request(url=f"{BASE_URL}/{dataset_name}")
    try:
        with urllib.request.urlopen(url_request) as url_file:
            url_content = url_file.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise DatasetNotFoundError(dataset_name) from e
        raise

    parser.feed(url_content)

    if parser.zip_name is None:
        raise ValueError(f"No zip file found for dataset '{dataset_name}'")

    return parser.zip_name


def _parse_info_string_value(value: str) -> Any:
    if value.lower() == "nan":
        return math.nan
    try:
        value = ast.literal_eval(value)
    except ValueError:
        pass

    return value


def _get_info_strings(comments: Sequence[str]) -> Mapping[str, Any]:

    info_strings_semicolons = {}
    info_strings_spaces = {}

    for comment in comments:
        if comment[0] not in {"-", "#"}:
            if re.fullmatch(INFO_STRING_SEMICOLONS_SEVERAL_REGEX, comment):
                for result in re.finditer(
                    INFO_STRING_SEMICOLONS_ONE_REGEX,
                    comment,
                ):
                    info_strings_semicolons[result.group(1)] = (
                        _parse_info_string_value(result.group(2))
                    )
            else:
                key, value = comment.rsplit(maxsplit=1)
                info_strings_spaces[key] = _parse_info_string_value(value)

    if info_strings_semicolons:
        return info_strings_semicolons

    # Check for absurd things in spaces
    if (
        len(info_strings_spaces) == 1
        or any(key.count(" ") > 3 for key in info_strings_spaces)
    ):
        return {}

    return info_strings_spaces


def _join_info_dicts(
    dicts: Sequence[Mapping[str, Any]],
) -> Mapping[str, np.typing.NDArray[Any]]:

    joined = {}

    n_keys = len(dicts[0])
    assert all(len(d) == n_keys for d in dicts)

    for key in dicts[0]:
        joined[key] = np.array([d[key] for d in dicts])

    return joined


def _constant_attrs(register: wfdb.Record) -> Sequence[Any]:
    return (register.n_sig, register.sig_name, register.units, register.fs)


@overload
def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
    target_column: str | Sequence[str] | None = None,
) -> Bunch:
    pass


@overload
def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
    target_column: str | Sequence[str] | None = None,
) -> Tuple[np.typing.NDArray[float], np.typing.NDArray[int]]:
    pass


@overload
def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: Literal[True],
    as_frame: Literal[True],
    target_column: str | Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    pass


def fetch(
    name: str,
    *,
    data_home: str | None = None,
    return_X_y: bool = False,
    as_frame: bool = False,
    target_column: str | Sequence[str] | None = None,
) -> (
    Bunch
    | Tuple[np.typing.NDArray[float], np.typing.NDArray[int]]
    | Tuple[pd.DataFrame, pd.Series | pd.DataFrame]
):

    zip_name = _get_zip_name_online(name)

    path = fetch_zip(
        dataname=name,
        urlname=f"{BASE_URL}/{name}/{zip_name}",
        subfolder="physionet",
        data_home=data_home,
    )

    subpath = path / Path(zip_name).stem
    if subpath.exists():
        path = subpath

    with open(path / "RECORDS") as records_file:
        records = [
            wfdb.io.rdrecord(str(path / record_name.rstrip('\n')))
            for record_name in records_file
        ]

    info_strings = [_get_info_strings(r.comments) for r in records]
    info = _join_info_dicts(info_strings)

    assert all(
        _constant_attrs(r) == _constant_attrs(records[0])
        for r in records
    )
    data = {
        "signal": [r.p_signal for r in records],
    }

    dataframe = pd.DataFrame(
        {**info, **data},
        index=[r.record_name for r in records],
    )
    dataframe["signal"].attrs.update(
        sig_name=records[0].sig_name,
        units=records[0].units,
        fs=records[0].fs,
    )

    data_dataframe = (
        dataframe
        if target_column is None
        else dataframe.drop(target_column, axis=1)
    )
    target_dataframe = (
        None
        if target_column is None
        else dataframe.loc[:, target_column]
    )

    data = (
        data_dataframe
        if as_frame is True
        else data_dataframe.to_numpy()
    )

    target = (
        None
        if target_dataframe is None
        else target_dataframe if as_frame is True
        else target_dataframe.to_numpy()
    )

    if return_X_y:
        return data, target

    # TODO: Fetch and download description from the website
    DESCR = ""

    feature_names = list(data_dataframe.keys())
    target_names = (
        None
        if target_dataframe is None
        else list(target_dataframe.keys())
    )

    bunch = Bunch(
        data=data,
        target=target,
        DESCR=DESCR,
        feature_names=feature_names,
        target_names=target_names,
    )

    if as_frame:
        bunch["frame"] = dataframe

    return bunch
