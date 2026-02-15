"""
한국 주식 유니버스, KRX ETF, 시장 카탈리스트 정의.

종목 분류:
  - large_cap:  시총 10조+ (삼성전자, SK하이닉스 등)
  - mid_large:  시총 1조~10조 (중형 우량주)
  - mid_cap:    시총 3000억~1조 (밸류+모멘텀 sweet spot)
  - small_mid:  시총 1000억~3000억 (고베타, 테마주)
"""

# =============================================================================
# 한국 주식 유니버스 (시가총액 1000억+ KRW)
# beta: KOSPI 대비 베타, us_corr: 미국 시장 연동성
# cap_tier: large_cap / mid_large / mid_cap / small_mid
# =============================================================================

KOREAN_STOCKS = {
    # =========================================================================
    # 대형주 (시총 10조+)
    # =========================================================================
    "005930": {"name": "Samsung Electronics", "sector": "Semiconductor", "beta": 1.1, "us_corr": "high", "cap_tier": "large_cap"},
    "000660": {"name": "SK Hynix", "sector": "Semiconductor", "beta": 1.3, "us_corr": "high", "cap_tier": "large_cap"},
    "373220": {"name": "LG Energy Solution", "sector": "Battery", "beta": 1.4, "us_corr": "high", "cap_tier": "large_cap"},
    "207940": {"name": "Samsung Biologics", "sector": "Bio/CDMO", "beta": 0.8, "us_corr": "medium", "cap_tier": "large_cap"},
    "005380": {"name": "Hyundai Motor", "sector": "Auto", "beta": 1.1, "us_corr": "medium", "cap_tier": "large_cap"},
    "000270": {"name": "Kia", "sector": "Auto", "beta": 1.2, "us_corr": "medium", "cap_tier": "large_cap"},
    "035420": {"name": "NAVER", "sector": "Internet/Platform", "beta": 1.0, "us_corr": "medium", "cap_tier": "large_cap"},
    "006400": {"name": "Samsung SDI", "sector": "Battery", "beta": 1.3, "us_corr": "high", "cap_tier": "large_cap"},
    "105560": {"name": "KB Financial Group", "sector": "Finance", "beta": 0.7, "us_corr": "low", "cap_tier": "large_cap"},
    "055550": {"name": "Shinhan Financial", "sector": "Finance", "beta": 0.7, "us_corr": "low", "cap_tier": "large_cap"},
    "003550": {"name": "LG", "sector": "Holding", "beta": 0.8, "us_corr": "low", "cap_tier": "large_cap"},

    # =========================================================================
    # 중대형주 (시총 1조~10조)
    # =========================================================================
    "035720": {"name": "Kakao", "sector": "Internet/Platform", "beta": 1.1, "us_corr": "medium", "cap_tier": "mid_large"},
    "051910": {"name": "LG Chem", "sector": "Chemical/Battery", "beta": 1.2, "us_corr": "medium", "cap_tier": "mid_large"},
    "068270": {"name": "Celltrion", "sector": "Bio/Biosimilar", "beta": 0.9, "us_corr": "medium", "cap_tier": "mid_large"},
    "086790": {"name": "Hana Financial", "sector": "Finance", "beta": 0.7, "us_corr": "low", "cap_tier": "mid_large"},
    "034730": {"name": "SK Inc", "sector": "Holding", "beta": 0.9, "us_corr": "low", "cap_tier": "mid_large"},
    "012330": {"name": "Hyundai Mobis", "sector": "Auto Parts", "beta": 0.9, "us_corr": "medium", "cap_tier": "mid_large"},
    "028260": {"name": "Samsung C&T", "sector": "Construction/Holding", "beta": 0.8, "us_corr": "low", "cap_tier": "mid_large"},
    "066570": {"name": "LG Electronics", "sector": "Electronics", "beta": 1.0, "us_corr": "medium", "cap_tier": "mid_large"},
    "005490": {"name": "POSCO Holdings", "sector": "Steel/Holding", "beta": 1.0, "us_corr": "medium", "cap_tier": "mid_large"},
    "032830": {"name": "Samsung Life", "sector": "Insurance", "beta": 0.6, "us_corr": "low", "cap_tier": "mid_large"},
    "009150": {"name": "Samsung Electro-Mechanics", "sector": "Electronic Components", "beta": 1.2, "us_corr": "high", "cap_tier": "mid_large"},
    "018260": {"name": "Samsung SDS", "sector": "IT Services", "beta": 0.8, "us_corr": "medium", "cap_tier": "mid_large"},
    "259960": {"name": "Krafton", "sector": "Gaming", "beta": 1.2, "us_corr": "medium", "cap_tier": "mid_large"},
    "033780": {"name": "KT&G", "sector": "Tobacco/Consumer", "beta": 0.4, "us_corr": "low", "cap_tier": "mid_large"},
    "017670": {"name": "SK Telecom", "sector": "Telecom", "beta": 0.5, "us_corr": "low", "cap_tier": "mid_large"},
    "030200": {"name": "KT", "sector": "Telecom", "beta": 0.5, "us_corr": "low", "cap_tier": "mid_large"},
    "015760": {"name": "Korea Electric Power", "sector": "Utility", "beta": 0.5, "us_corr": "low", "cap_tier": "mid_large"},
    "010130": {"name": "Korea Zinc", "sector": "Non-ferrous Metals", "beta": 0.9, "us_corr": "medium", "cap_tier": "mid_large"},
    "352820": {"name": "HYBE", "sector": "Entertainment", "beta": 1.3, "us_corr": "low", "cap_tier": "mid_large"},
    "012450": {"name": "Hanwha Aerospace", "sector": "Defense/Aerospace", "beta": 1.4, "us_corr": "medium", "cap_tier": "mid_large"},
    "047810": {"name": "Korea Aerospace Industries", "sector": "Defense/Aerospace", "beta": 1.3, "us_corr": "medium", "cap_tier": "mid_large"},
    "267250": {"name": "HD Hyundai", "sector": "Shipbuilding/Holding", "beta": 1.1, "us_corr": "low", "cap_tier": "mid_large"},
    "009540": {"name": "HD Korea Shipbuilding", "sector": "Shipbuilding", "beta": 1.3, "us_corr": "low", "cap_tier": "mid_large"},
    "329180": {"name": "HD Hyundai Heavy", "sector": "Shipbuilding", "beta": 1.4, "us_corr": "low", "cap_tier": "mid_large"},
    "034020": {"name": "Doosan Enerbility", "sector": "Nuclear/Energy", "beta": 1.5, "us_corr": "medium", "cap_tier": "mid_large"},

    # =========================================================================
    # 중형주 (시총 3000억~1조) - 밸류 + 모멘텀 Sweet Spot
    # =========================================================================

    # 반도체 장비/소재
    "042700": {"name": "Hanmi Semiconductor", "sector": "Semiconductor Equipment", "beta": 1.6, "us_corr": "high", "cap_tier": "mid_cap"},
    "240810": {"name": "Wonik IPS", "sector": "Semiconductor Equipment", "beta": 1.5, "us_corr": "high", "cap_tier": "mid_cap"},
    "095340": {"name": "ISC", "sector": "Semiconductor Test", "beta": 1.4, "us_corr": "high", "cap_tier": "mid_cap"},
    "302920": {"name": "Leeno Industrial", "sector": "Semiconductor Probe", "beta": 1.3, "us_corr": "high", "cap_tier": "mid_cap"},
    "357780": {"name": "Solbrain", "sector": "Semiconductor Chemical", "beta": 1.2, "us_corr": "high", "cap_tier": "mid_cap"},
    "036830": {"name": "Soulbrain Holdings", "sector": "Semiconductor Chemical", "beta": 1.1, "us_corr": "high", "cap_tier": "mid_cap"},
    "166090": {"name": "Hana Materials", "sector": "Semiconductor Parts", "beta": 1.4, "us_corr": "high", "cap_tier": "mid_cap"},
    "403870": {"name": "HPSP", "sector": "Semiconductor Equipment", "beta": 1.7, "us_corr": "high", "cap_tier": "mid_cap"},
    "058470": {"name": "Risu Media", "sector": "Semiconductor Substrate", "beta": 1.5, "us_corr": "high", "cap_tier": "mid_cap"},
    "185750": {"name": "Jusung Engineering", "sector": "Semiconductor Equipment", "beta": 1.6, "us_corr": "high", "cap_tier": "mid_cap"},

    # 2차전지 소재/부품
    "247540": {"name": "Ecopro BM", "sector": "Battery Cathode", "beta": 1.8, "us_corr": "high", "cap_tier": "mid_cap"},
    "086520": {"name": "Ecopro", "sector": "Battery Materials Holding", "beta": 1.9, "us_corr": "high", "cap_tier": "mid_cap"},
    "066970": {"name": "L&F", "sector": "Battery Cathode", "beta": 1.8, "us_corr": "high", "cap_tier": "mid_cap"},
    "003670": {"name": "POSCO Future M", "sector": "Battery Anode/Cathode", "beta": 1.5, "us_corr": "high", "cap_tier": "mid_cap"},
    "336370": {"name": "Solus Advanced Materials", "sector": "Battery Copper Foil", "beta": 1.6, "us_corr": "high", "cap_tier": "mid_cap"},
    "298040": {"name": "Hyosung Heavy Industries", "sector": "Transformer/Grid", "beta": 1.3, "us_corr": "medium", "cap_tier": "mid_cap"},
    "281820": {"name": "Keum Yang", "sector": "Battery Housing", "beta": 1.5, "us_corr": "medium", "cap_tier": "mid_cap"},
    "064350": {"name": "Hyundai Rotem", "sector": "Defense/Rail", "beta": 1.4, "us_corr": "low", "cap_tier": "mid_cap"},

    # 바이오/제약
    "028300": {"name": "HLB", "sector": "Bio/Oncology", "beta": 2.0, "us_corr": "medium", "cap_tier": "mid_cap"},
    "000100": {"name": "Yuhan", "sector": "Pharma", "beta": 0.8, "us_corr": "medium", "cap_tier": "mid_cap"},
    "145020": {"name": "Hugel", "sector": "Bio/Aesthetics", "beta": 1.2, "us_corr": "medium", "cap_tier": "mid_cap"},
    "326030": {"name": "SK Biopharmaceuticals", "sector": "Pharma/CNS", "beta": 1.1, "us_corr": "medium", "cap_tier": "mid_cap"},
    "196170": {"name": "Alteogen", "sector": "Bio/Platform", "beta": 1.5, "us_corr": "medium", "cap_tier": "mid_cap"},
    "268600": {"name": "Samsung Vaccine", "sector": "Bio/Vaccine", "beta": 1.3, "us_corr": "medium", "cap_tier": "mid_cap"},
    "141080": {"name": "Regen Biotech", "sector": "Bio/Stem Cell", "beta": 1.7, "us_corr": "low", "cap_tier": "mid_cap"},

    # 방산
    "014880": {"name": "LIG Nex1", "sector": "Defense/Missile", "beta": 1.2, "us_corr": "medium", "cap_tier": "mid_cap"},
    "012750": {"name": "Hanwha Systems", "sector": "Defense/IT", "beta": 1.3, "us_corr": "medium", "cap_tier": "mid_cap"},

    # 전력기기/신에너지
    "010120": {"name": "LS Electric", "sector": "Electrical Equipment", "beta": 1.3, "us_corr": "medium", "cap_tier": "mid_cap"},
    "267260": {"name": "HD Hyundai Electric", "sector": "Transformer/Switchgear", "beta": 1.5, "us_corr": "medium", "cap_tier": "mid_cap"},
    "103590": {"name": "Ilhin Electric", "sector": "Transformer", "beta": 1.6, "us_corr": "medium", "cap_tier": "mid_cap"},
    "071970": {"name": "STX Heavy Industries", "sector": "Generator/Engine", "beta": 1.7, "us_corr": "low", "cap_tier": "mid_cap"},

    # 게임/엔터
    "036570": {"name": "NCsoft", "sector": "Gaming", "beta": 1.1, "us_corr": "medium", "cap_tier": "mid_cap"},
    "293490": {"name": "Kakao Games", "sector": "Gaming", "beta": 1.3, "us_corr": "medium", "cap_tier": "mid_cap"},
    "263750": {"name": "Pearl Abyss", "sector": "Gaming", "beta": 1.4, "us_corr": "medium", "cap_tier": "mid_cap"},
    "251270": {"name": "Netmarble", "sector": "Gaming/Mobile", "beta": 1.2, "us_corr": "medium", "cap_tier": "mid_cap"},
    "112040": {"name": "Wemade", "sector": "Gaming/Blockchain", "beta": 1.6, "us_corr": "medium", "cap_tier": "mid_cap"},

    # 소비재/유통
    "090430": {"name": "Amorepacific", "sector": "Cosmetics", "beta": 1.0, "us_corr": "low", "cap_tier": "mid_cap"},
    "097950": {"name": "CJ CheilJedang", "sector": "Food/Bio", "beta": 0.7, "us_corr": "low", "cap_tier": "mid_cap"},
    "004170": {"name": "Shinsegae", "sector": "Retail/Department", "beta": 0.9, "us_corr": "low", "cap_tier": "mid_cap"},
    "069960": {"name": "Hyundai Department Store", "sector": "Retail/Department", "beta": 0.8, "us_corr": "low", "cap_tier": "mid_cap"},

    # 산업재/건설
    "006360": {"name": "GS E&C", "sector": "Construction", "beta": 1.1, "us_corr": "low", "cap_tier": "mid_cap"},
    "000720": {"name": "Hyundai E&C", "sector": "Construction", "beta": 1.0, "us_corr": "low", "cap_tier": "mid_cap"},
    "047050": {"name": "POSCO International", "sector": "Trading/LNG", "beta": 1.0, "us_corr": "medium", "cap_tier": "mid_cap"},
    "003490": {"name": "Korean Air", "sector": "Airline", "beta": 1.3, "us_corr": "medium", "cap_tier": "mid_cap"},
    "020560": {"name": "Asiana Airlines", "sector": "Airline", "beta": 1.4, "us_corr": "medium", "cap_tier": "mid_cap"},

    # IT/소프트웨어
    "035760": {"name": "CJ ENM", "sector": "Media/Content", "beta": 1.2, "us_corr": "low", "cap_tier": "mid_cap"},
    "041510": {"name": "SM Entertainment", "sector": "K-Pop/Entertainment", "beta": 1.4, "us_corr": "low", "cap_tier": "mid_cap"},

    # =========================================================================
    # 소형~중형 (시총 1000억~3000억) - 고베타, 테마주
    # =========================================================================
    "383310": {"name": "EcoproHN", "sector": "Battery Recycling", "beta": 2.0, "us_corr": "high", "cap_tier": "small_mid"},
    "950160": {"name": "Koltec", "sector": "Nuclear Equipment", "beta": 1.8, "us_corr": "low", "cap_tier": "small_mid"},
    "322000": {"name": "Hyundai Energy Solutions", "sector": "Solar Module", "beta": 1.7, "us_corr": "high", "cap_tier": "small_mid"},
    "417200": {"name": "Telechips", "sector": "Auto Semiconductor", "beta": 1.6, "us_corr": "high", "cap_tier": "small_mid"},
    "039030": {"name": "IOK", "sector": "Media/Content", "beta": 1.5, "us_corr": "low", "cap_tier": "small_mid"},
    "192820": {"name": "Cosmos Pharma", "sector": "CDMO/Pharma", "beta": 1.4, "us_corr": "medium", "cap_tier": "small_mid"},
    "200710": {"name": "Danaher Korea (Conceptia)", "sector": "Bio Equipment", "beta": 1.3, "us_corr": "medium", "cap_tier": "small_mid"},
    "018290": {"name": "POSCO DX", "sector": "IT Services/AI", "beta": 1.5, "us_corr": "medium", "cap_tier": "small_mid"},
    "138930": {"name": "BNK Financial", "sector": "Regional Finance", "beta": 0.6, "us_corr": "low", "cap_tier": "small_mid"},
    "139130": {"name": "DGB Financial", "sector": "Regional Finance", "beta": 0.6, "us_corr": "low", "cap_tier": "small_mid"},
    "161390": {"name": "Hankook Tire", "sector": "Tire", "beta": 0.8, "us_corr": "low", "cap_tier": "small_mid"},
    "011170": {"name": "Lotte Chemical", "sector": "Petrochemical", "beta": 1.1, "us_corr": "medium", "cap_tier": "small_mid"},
    "006800": {"name": "Mirae Asset Securities", "sector": "Securities", "beta": 1.2, "us_corr": "medium", "cap_tier": "small_mid"},
    "016360": {"name": "Samsung Securities", "sector": "Securities", "beta": 1.0, "us_corr": "medium", "cap_tier": "small_mid"},
}


# =============================================================================
# KRX 상장 해외 연동 ETF
# =============================================================================

KRX_ETFS = {
    "long_us": [
        {"name": "KODEX S&P500", "ticker": "379800"},
        {"name": "TIGER NASDAQ100", "ticker": "133690"},
        {"name": "KODEX NASDAQ100", "ticker": "304940"},
        {"name": "TIGER S&P500", "ticker": "360750"},
        {"name": "TIGER 미국필라델피아반도체", "ticker": "381180"},
    ],
    "inverse_us": [
        {"name": "KODEX S&P500 Inverse", "ticker": "251340"},
        {"name": "TIGER S&P500 Inverse", "ticker": "225030"},
        {"name": "KODEX NASDAQ100 Inverse", "ticker": "368190"},
    ],
    "short_kospi": [
        {"name": "KODEX 200 Inverse", "ticker": "114800"},
        {"name": "KODEX 200 Inverse 2X", "ticker": "252670"},
        {"name": "TIGER 200 Inverse", "ticker": "123310"},
    ],
    "leverage_kospi": [
        {"name": "KODEX 레버리지", "ticker": "122630"},
        {"name": "TIGER 200 선물레버리지", "ticker": "253250"},
    ],
}


# =============================================================================
# Regime 정의
# =============================================================================

REGIMES = ["STRONG_BULL", "MILD_BULL", "WEAKENING", "SHORT_TERM_STRESS", "BEAR"]


# =============================================================================
# US 시장 카탈리스트 (Overnight Gap 전략용)
# =============================================================================

US_CATALYSTS_BULLISH = [
    "NVIDIA earnings beat expectations with strong data center revenue guidance",
    "Fed signals potential rate cut in upcoming meeting, dovish pivot",
    "US CPI came in below expectations, easing inflation concerns",
    "S&P 500 broke all-time high with strong breadth",
    "US jobs report strong but wages moderate, goldilocks scenario",
    "Apple announced record iPhone sales in Asia markets",
    "Tesla reported strong delivery numbers, EV demand resilient",
    "US-China trade tensions eased after diplomatic meeting",
    "US GDP growth exceeded consensus, soft landing narrative intact",
    "Microsoft Azure revenue surged 30%+, AI capex cycle continues",
    "US retail sales beat expectations, consumer remains strong",
    "Semiconductor equipment orders surged, capex upcycle confirmed",
    "Meta announced massive AI infrastructure investment plan",
    "Broadcom raised guidance on AI networking chip demand",
    "TSMC monthly revenue hit record high, AI demand sustained",
]

US_CATALYSTS_BEARISH = [
    "Fed chair signaled rates will stay higher for longer",
    "US CPI exceeded expectations, sticky inflation concerns",
    "US 10-year Treasury yield surged above 5%",
    "Major US bank reported higher-than-expected loan loss provisions",
    "Geopolitical escalation in Middle East, oil prices spiked",
    "US unemployment claims jumped significantly",
    "US consumer confidence index dropped sharply",
    "China economic data disappointing, global growth concerns",
    "US debt ceiling crisis re-emerged, government shutdown risk",
    "Major tech company missed earnings, AI monetization questioned",
    "US housing market data collapsed, mortgage rates at 8%",
    "Trade war escalation: new tariffs announced on key imports",
    "Japan BOJ surprise rate hike triggered global carry trade unwind",
    "VIX spiked above 30, systematic selling accelerated",
    "US regional bank concerns resurfaced, deposit outflows reported",
]


# =============================================================================
# 텔레그램/뉴스 테마 (한국 시장 특화)
# =============================================================================

TELEGRAM_THEMES_BULLISH = [
    "battery sector rally driven by US IRA subsidy confirmation",
    "semiconductor stocks trending after SK Hynix HBM order news",
    "auto stocks surging on Hyundai-GM EV platform partnership rumor",
    "bio sector momentum after Celltrion FDA approval news",
    "retail investors piling into battery material stocks",
    "AI-related Korean stocks gaining attention after US tech earnings",
    "defense stocks trending on increased military budget news",
    "institutional buying streak in financial sector continues",
    "transformer/grid stocks surging on US power grid investment news",
    "shipbuilding stocks rallying on record LNG carrier orders",
    "Hanmi Semiconductor trending after HBM test equipment order surge",
    "nuclear power stocks in play after government policy reversal",
    "K-pop/entertainment stocks rallying on global tour revenue records",
    "POSCO DX trending on smart factory AI deal with major manufacturer",
    "Korean cosmetics stocks surging on China travel recovery news",
]

TELEGRAM_THEMES_BEARISH = [
    "short-selling concerns in battery sector after EV demand slowdown",
    "negative sentiment on semiconductor cycle downturn warnings",
    "credit risk concerns for Korean construction companies",
    "foreign investors accelerating sell-off in Korean equities",
    "margin call fears as KOSDAQ stocks plunge",
    "regulatory crackdown fears on platform companies",
    "China recession fears impacting Korean export stocks",
    "won weakening past 1400/USD raising import cost concerns",
    "bio stocks crash after clinical trial failure news",
    "gaming stocks slump on new regulation concerns",
    "EV subsidy cut fears hitting battery supply chain stocks",
    "construction sector concerns after PF (project financing) defaults",
    "airline stocks dropping on fuel cost spike",
    "retail sector weakness on declining consumer spending data",
    "secondary battery stocks plunging on oversupply warnings",
]
