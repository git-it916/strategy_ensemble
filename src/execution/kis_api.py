"""
KIS (Korea Investment & Securities) API Wrapper

한국투자증권 Open API 래퍼
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
import logging

import requests

logger = logging.getLogger(__name__)

# API Endpoints
KIS_BASE_URL_REAL = "https://openapi.koreainvestment.com:9443"
KIS_BASE_URL_PAPER = "https://openapivts.koreainvestment.com:29443"


@dataclass
class KISAuth:
    """KIS API authentication info."""
    app_key: str
    app_secret: str
    account_number: str  # 계좌번호 (8자리-2자리)
    account_product_code: str = "01"  # 계좌상품코드
    is_paper: bool = True  # 모의투자 여부
    customer_type: str = "P"  # P: personal, B: corporate
    exchange_id: str = "KRX"  # KRX / NXT / SOR / ALL

    @property
    def base_url(self) -> str:
        return KIS_BASE_URL_PAPER if self.is_paper else KIS_BASE_URL_REAL


class KISApi:
    """
    한국투자증권 Open API 래퍼.

    Features:
        - OAuth 토큰 자동 갱신
        - 주식 시세 조회
        - 매수/매도 주문
        - 잔고 조회
        - 체결 내역 조회
    """

    def __init__(self, auth: KISAuth):
        """
        Initialize KIS API.

        Args:
            auth: Authentication info
        """
        self.auth = auth
        self._access_token: str | None = None
        self._token_expires: datetime | None = None

    def _get_headers(
        self,
        tr_id: str,
        tr_cont: str = "",
        hash_body: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Get request headers with auth token."""
        if self._access_token is None or self._is_token_expired():
            self._refresh_token()

        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._access_token}",
            "appkey": self.auth.app_key,
            "appsecret": self.auth.app_secret,
            "tr_id": tr_id,
            "custtype": self.auth.customer_type,
        }
        if tr_cont:
            headers["tr_cont"] = tr_cont
        if hash_body is not None:
            headers["hashkey"] = self._get_hashkey(hash_body, headers)
        return headers

    def _is_token_expired(self) -> bool:
        """Check if token is expired."""
        if self._token_expires is None:
            return True
        return datetime.now() >= self._token_expires - timedelta(minutes=5)

    def _refresh_token(self) -> None:
        """Refresh OAuth token."""
        url = f"{self.auth.base_url}/oauth2/tokenP"

        data = {
            "grant_type": "client_credentials",
            "appkey": self.auth.app_key,
            "appsecret": self.auth.app_secret,
        }

        response = requests.post(url, json=data)
        response.raise_for_status()

        result = response.json()
        self._access_token = result["access_token"]

        expires_at = result.get("access_token_token_expired")
        if expires_at:
            try:
                self._token_expires = datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                self._token_expires = datetime.now() + timedelta(hours=23)
        else:
            self._token_expires = datetime.now() + timedelta(hours=23)
        logger.info("KIS access token refreshed")

    def _get_hashkey(self, data: dict[str, Any], headers: dict[str, str]) -> str:
        """Get hashkey for POST requests."""
        url = f"{self.auth.base_url}/uapi/hashkey"
        hash_headers = {
            k: v for k, v in headers.items() if k not in {"tr_id", "tr_cont", "hashkey"}
        }
        hash_headers["content-type"] = "application/json; charset=utf-8"

        response = requests.post(url, headers=hash_headers, data=json.dumps(data))
        response.raise_for_status()
        payload = response.json()
        hashkey = payload.get("HASH")
        if not hashkey:
            raise Exception(f"Failed to get hashkey: {payload}")
        return hashkey

    def get_price(self, stock_code: str) -> dict[str, Any]:
        """
        현재가 조회.

        Args:
            stock_code: 종목코드 (예: "005930")

        Returns:
            현재가 정보
        """
        url = f"{self.auth.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"

        # 실전/모의 TR_ID
        tr_id = "FHKST01010100" if not self.auth.is_paper else "FHKST01010100"

        headers = self._get_headers(tr_id)

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 주식
            "FID_INPUT_ISCD": stock_code,
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("rt_cd") != "0":
            raise Exception(f"API Error: {data.get('msg1')}")

        output = data.get("output", {})

        return {
            "stock_code": stock_code,
            "name": output.get("hts_kor_isnm", ""),
            "price": int(output.get("stck_prpr", 0)),
            "change": int(output.get("prdy_vrss", 0)),
            "change_rate": float(output.get("prdy_ctrt", 0)),
            "volume": int(output.get("acml_vol", 0)),
            "high": int(output.get("stck_hgpr", 0)),
            "low": int(output.get("stck_lwpr", 0)),
            "open": int(output.get("stck_oprc", 0)),
        }

    def get_balance(self) -> dict[str, Any]:
        """
        잔고 조회.

        Returns:
            계좌 잔고 정보
        """
        url = f"{self.auth.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"

        tr_id = "VTTC8434R" if self.auth.is_paper else "TTTC8434R"
        headers = self._get_headers(tr_id)

        params = {
            "CANO": self.auth.account_number[:8],
            "ACNT_PRDT_CD": self.auth.account_number[8:] or self.auth.account_product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("rt_cd") != "0":
            raise Exception(f"API Error: {data.get('msg1')}")

        output1 = data.get("output1", [])  # 보유종목
        output2 = data.get("output2", [{}])[0]  # 계좌요약

        holdings = []
        for item in output1:
            holdings.append({
                "stock_code": item.get("pdno"),
                "name": item.get("prdt_name"),
                "quantity": int(item.get("hldg_qty", 0)),
                "avg_price": float(item.get("pchs_avg_pric", 0)),
                "current_price": int(item.get("prpr", 0)),
                "eval_amount": int(item.get("evlu_amt", 0)),
                "profit_loss": int(item.get("evlu_pfls_amt", 0)),
                "profit_rate": float(item.get("evlu_pfls_rt", 0)),
            })

        return {
            "holdings": holdings,
            "total_eval": int(output2.get("tot_evlu_amt", 0)),
            "cash": int(output2.get("dnca_tot_amt", 0)),
            "total_profit_loss": int(output2.get("evlu_pfls_smtl_amt", 0)),
        }

    def buy_stock(
        self,
        stock_code: str,
        quantity: int,
        price: int | None = None,
        order_type: str = "00",  # 00: 지정가, 01: 시장가
        excg_id_dvsn_cd: str | None = None,
        sll_type: str = "",
        cndt_pric: str = "",
    ) -> dict[str, Any]:
        """
        매수 주문.

        Args:
            stock_code: 종목코드
            quantity: 수량
            price: 가격 (시장가일 경우 0)
            order_type: 주문유형

        Returns:
            주문 결과
        """
        return self._place_order(
            stock_code=stock_code,
            quantity=quantity,
            price=price or 0,
            order_type=order_type,
            is_buy=True,
            excg_id_dvsn_cd=excg_id_dvsn_cd,
            sll_type=sll_type,
            cndt_pric=cndt_pric,
        )

    def sell_stock(
        self,
        stock_code: str,
        quantity: int,
        price: int | None = None,
        order_type: str = "00",
        excg_id_dvsn_cd: str | None = None,
        sll_type: str = "",
        cndt_pric: str = "",
    ) -> dict[str, Any]:
        """
        매도 주문.

        Args:
            stock_code: 종목코드
            quantity: 수량
            price: 가격
            order_type: 주문유형

        Returns:
            주문 결과
        """
        return self._place_order(
            stock_code=stock_code,
            quantity=quantity,
            price=price or 0,
            order_type=order_type,
            is_buy=False,
            excg_id_dvsn_cd=excg_id_dvsn_cd,
            sll_type=sll_type,
            cndt_pric=cndt_pric,
        )

    def _place_order(
        self,
        stock_code: str,
        quantity: int,
        price: int,
        order_type: str,
        is_buy: bool,
        excg_id_dvsn_cd: str | None = None,
        sll_type: str = "",
        cndt_pric: str = "",
    ) -> dict[str, Any]:
        """Place buy/sell order."""
        url = f"{self.auth.base_url}/uapi/domestic-stock/v1/trading/order-cash"

        # TR_ID for buy/sell, paper/real
        if self.auth.is_paper:
            tr_id = "VTTC0012U" if is_buy else "VTTC0011U"
        else:
            tr_id = "TTTC0012U" if is_buy else "TTTC0011U"

        data = {
            "CANO": self.auth.account_number[:8],
            "ACNT_PRDT_CD": self.auth.account_number[8:] or self.auth.account_product_code,
            "PDNO": stock_code,
            "ORD_DVSN": order_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
            "EXCG_ID_DVSN_CD": excg_id_dvsn_cd or self.auth.exchange_id,
        }
        if not is_buy and sll_type:
            data["SLL_TYPE"] = sll_type
        if cndt_pric:
            data["CNDT_PRIC"] = cndt_pric

        headers = self._get_headers(tr_id, hash_body=data)

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()

        result = response.json()

        if result.get("rt_cd") != "0":
            raise Exception(f"Order failed: {result.get('msg1')}")

        output = result.get("output", {})

        logger.info(
            f"{'BUY' if is_buy else 'SELL'} order placed: "
            f"{stock_code} x {quantity} @ {price}"
        )

        return {
            "order_no": output.get("ODNO"),
            "order_time": output.get("ORD_TMD"),
            "krx_fwdg_ord_orgno": output.get("KRX_FWDG_ORD_ORGNO"),
            "stock_code": stock_code,
            "quantity": quantity,
            "price": price,
            "is_buy": is_buy,
        }

    def get_orders(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str = "inner",
        excg_id_dvsn_cd: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        당일 주문 내역 조회.

        Returns:
            주문 내역 리스트
        """
        url = f"{self.auth.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"

        if self.auth.is_paper:
            tr_id = "VTTC0081R" if period == "inner" else "VTSC9215R"
        else:
            tr_id = "TTTC0081R" if period == "inner" else "CTSC9215R"
        headers = self._get_headers(tr_id)

        if start_date is None:
            start_date = datetime.now().strftime("%Y%m%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        params = {
            "CANO": self.auth.account_number[:8],
            "ACNT_PRDT_CD": self.auth.account_number[8:] or self.auth.account_product_code,
            "INQR_STRT_DT": start_date,
            "INQR_END_DT": end_date,
            "SLL_BUY_DVSN_CD": "00",  # 전체
            "PDNO": "",
            "CCLD_DVSN": "00",
            "INQR_DVSN": "00",
            "INQR_DVSN_3": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        if excg_id_dvsn_cd or self.auth.exchange_id:
            params["EXCG_ID_DVSN_CD"] = excg_id_dvsn_cd or self.auth.exchange_id

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("rt_cd") != "0":
            raise Exception(f"API Error: {data.get('msg1')}")

        orders = []
        for item in data.get("output1", []):
            orders.append({
                "order_no": item.get("odno"),
                "stock_code": item.get("pdno"),
                "stock_name": item.get("prdt_name"),
                "order_type": "BUY" if item.get("sll_buy_dvsn_cd") == "02" else "SELL",
                "order_qty": int(item.get("ord_qty", 0)),
                "order_price": int(item.get("ord_unpr", 0)),
                "filled_qty": int(item.get("tot_ccld_qty", 0)),
                "filled_price": int(item.get("avg_prvs", 0)),
                "order_time": item.get("ord_tmd"),
                "order_orgno": item.get("ord_orgno"),
                "order_branch_no": item.get("ord_gno_brno"),
                "org_order_no": item.get("orgn_odno"),
                "cancel_yn": item.get("cncl_yn"),
                "remain_qty": int(item.get("rmn_qty", 0)),
                "reject_qty": int(item.get("rjct_qty", 0)),
            })

        return orders

    def cancel_order(
        self,
        order_no: str,
        stock_code: str,
        quantity: int,
        order_type: str = "00",
        krx_fwdg_ord_orgno: str = "",
        excg_id_dvsn_cd: str | None = None,
        qty_all_ord_yn: str = "Y",
        cndt_pric: str = "",
    ) -> dict[str, Any]:
        """
        주문 취소.

        Args:
            order_no: 주문번호
            stock_code: 종목코드
            quantity: 취소 수량

        Returns:
            취소 결과
        """
        url = f"{self.auth.base_url}/uapi/domestic-stock/v1/trading/order-rvsecncl"

        tr_id = "VTTC0013U" if self.auth.is_paper else "TTTC0013U"

        data = {
            "CANO": self.auth.account_number[:8],
            "ACNT_PRDT_CD": self.auth.account_number[8:] or self.auth.account_product_code,
            "KRX_FWDG_ORD_ORGNO": krx_fwdg_ord_orgno,
            "ORGN_ODNO": order_no,
            "ORD_DVSN": order_type,
            "RVSE_CNCL_DVSN_CD": "02",  # 취소
            "ORD_QTY": str(quantity),
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": qty_all_ord_yn,
            "EXCG_ID_DVSN_CD": excg_id_dvsn_cd or self.auth.exchange_id,
        }
        if cndt_pric:
            data["CNDT_PRIC"] = cndt_pric

        headers = self._get_headers(tr_id, hash_body=data)

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()

        result = response.json()

        if result.get("rt_cd") != "0":
            raise Exception(f"Cancel failed: {result.get('msg1')}")

        return {
            "status": "cancelled",
            "order_no": order_no,
        }
