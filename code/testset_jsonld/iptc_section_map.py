"""Mapping from publisher section names (JSON-LD articleSection / breadcrumb) to the 17 top-level
IPTC Media Topics, grounded in the official IPTC Media Topic tree.

Every group of section names is tied to one IPTC Media Topic node (by its en-GB name). At import
the node is checked against the official tree (iptc_reference/mediatopic_en-GB.json, parsed from
https://www.iptc.org/std/NewsCodes/treeview/mediatopic/mediatopic-en-GB.html, version 2026-07-09):
it must exist, must not be retired, and its level-1 ancestor must be the label it maps to.

Precision rule: a section name is mapped only if, according to the IPTC tree, its content belongs
to a single top-level topic. Names whose content IPTC places under two top-level topics are listed
in AMBIGUOUS with the two nodes and are never mapped, e.g.
  "tech"      -> science and technology > technology and engineering
                 economy, business and finance > products and services > computing and IT
  "terrorism" -> conflict, war and peace > act of terror  |  crime, law and justice > crime > terrorism
Mixed "news of the day" sections (cronaca, sucesos, güncel, общество, ...) and generic or
geographic sections (news, world, local, ...) are not mapped either.

Names are matched on the normalised full string (see normalise), never as substrings.

    python testset_jsonld/iptc_section_map.py     # validate and export iptc_reference/mapping_table.csv
"""
import json
import re
import unicodedata
from pathlib import Path

LABELS = [
    "arts, culture, entertainment and media", "conflict, war and peace", "crime, law and justice",
    "disaster, accident and emergency incident", "economy, business and finance", "education",
    "environment", "health", "human interest", "labour", "lifestyle and leisure", "politics",
    "religion", "science and technology", "society", "sport", "weather",
]
# Our label names are Kuzman & Ljubesic's; IPTC's own level-1 name differs only for politics.
IPTC_TOP = {label: label for label in LABELS} | {"politics": "politics and government"}

REFERENCE = Path(__file__).with_name("iptc_reference") / "mediatopic_en-GB.json"

# label -> {IPTC node name: section names}
MAPPING: dict[str, dict[str, set[str]]] = {
    "arts, culture, entertainment and media": {
        "arts, culture, entertainment and media": {
            "culture", "arts", "art", "arts & culture", "arts and culture", "arts & entertainment", "cultura",
            "kultur", "kultura", "culture & arts", "budaya", "seni", "seni budaya", "kebudayaan", "kültür",
            "kültür sanat", "kültür-sanat", "kültür - sanat", "kültür sanat haberleri", "sanat", "культура",
            "文化", "ثقافة", "الثقافة", "فرهنگ", "संस्कृति", "문화", "văn hóa", "วัฒนธรรม", "πολιτισμός",
            "kultur & kunst", "art & design"},
        "music": {"music", "musik", "música", "musica", "musique", "müzik", "музыка", "音乐", "موسيقى", "संगीत", "음악"},
        "cinema": {"movies", "movie", "film", "films", "cinema", "cine", "kino", "sinema", "кино", "电影", "سينما",
                   "फिल्म", "영화"},
        "literature": {"books", "book", "literature", "libros", "livres", "bücher", "buku", "sastra", "literatura",
                       "kitap"},
        "television": {"tv", "television", "televisión", "televisione", "televisi", "tv & radio", "tv and radio"},
        "theatre": {"theatre", "theater", "teatro", "stage"},
        "fashion": {"fashion", "moda", "mode"},
        "mass media": {"media", "medios"},
        "library and museum": {"museums"},
    },
    "conflict, war and peace": {
        "war": {"war", "war in ukraine", "ukraine war", "russia-ukraine war", "israel-hamas war", "israel-gaza war",
                "gaza war", "perang", "guerra", "guerre", "krieg", "wojna", "savaş", "война", "війна", "战争", "حرب",
                "الحرب", "जंग", "युद्ध", "전쟁", "chiến tranh", "military conflict", "war & peace", "conflict",
                "conflicts"},
    },
    "crime, law and justice": {
        "crime, law and justice": {
            "crime", "crimes", "crime & courts", "crime and courts", "law & order", "justice", "kriminal",
            "kriminalitas", "jenayah", "crimen", "criminalità", "kriminalität", "przestępczość", "криминал", "犯罪",
            "جريمة", "الجريمة", "अपराध", "क्राइम", "อาชญากรรม", "true crime", "hukum & kriminal",
            "hukum dan kriminal", "justice & police"},
        "court": {"courts", "court", "tribunales", "tribunale", "tribunais", "mahkamah", "adliye", "суд", "judicial",
                  "judiciales", "yargı", "قضاء"},
        "law": {"law", "legal", "hukum", "право", "法律", "法治", "pháp luật", "giustizia", "justicia", "justiça", "justiz"},
        "police": {"police", "polisi", "policía", "policia", "polizei", "policiales", "policial", "asayiş"},
    },
    "disaster, accident and emergency incident": {
        "disaster, accident and emergency incident": {
            "disaster", "disasters", "bencana", "desastres", "catástrofes", "katastrofy", "катастрофы", "灾害",
            "كوارث", "재난", "afet", "emergencies", "catastrophes"},
        "natural disaster": {"natural disasters", "bencana alam", "thiên tai", "deprem", "earthquake", "gempa"},
        "fire": {"fires", "wildfires", "kebakaran"},
        "accident and emergency incident": {"accidents", "accident", "kecelakaan", "unfälle", "wypadki", "kazalar",
                                            "аварии", "事故", "दुर्घटना", "tai nạn", "incidenti"},
        "road accident and incident": {"дтп", "حوادث الطرق"},
    },
    "economy, business and finance": {
        "economy, business and finance": {
            "business", "economy", "finance", "finances", "money", "business news", "economy news", "bisnis",
            "ekonomi", "ekonomi bisnis", "ekonomi dan bisnis", "keuangan", "finansial", "ekbis", "economía",
            "economia", "économie", "wirtschaft", "gospodarka", "ekonomika", "talous", "negocios", "negócios",
            "negocio", "empresas", "finanzas", "finanças", "finance & economy", "экономика", "бизнес", "финансы",
            "економіка", "经济", "經濟", "财经", "財經", "商业", "اقتصاد", "الاقتصاد", "اقتصادی", "अर्थव्यवस्था",
            "बिज़नेस", "व्यापार", "경제", "経済", "kinh tế", "kinh doanh", "เศรษฐกิจ", "οικονομία", "biznes", "finanza",
            "finanzen", "companies", "industry", "ቢዝነስ", "uchumi"},
        "market and exchange": {"markets", "mercados", "markets news", "borsa", "bolsa", "giełda", "börse",
                                "stock market", "stocks", "aktien", "saham", "pasar modal", "capital market news"},
        "financial service": {"banking", "perbankan", "personal finance"},
        "construction and property": {"real estate", "properti", "property"},
        "cryptocurrency": {"crypto", "cryptocurrency", "kripto"},
        "agriculture": {"agriculture", "agribusiness", "pertanian", "agricultura", "agricoltura", "tarım",
                        "сельское хозяйство", "农业", "الزراعة", "कृषि"},
        "energy and resource": {"energy business"},
    },
    "education": {
        "education": {"education", "pendidikan", "edukasi", "educación", "educação", "éducation", "istruzione",
                      "bildung", "edukacja", "eğitim", "образование", "освіта", "教育", "تعليم", "التعليم", "آموزش",
                      "शिक्षा", "교육", "giáo dục", "การศึกษา", "εκπαίδευση", "παιδεία"},
        "school": {"schools", "school", "scuola", "k-12"},
        "college and university": {"higher education", "universities", "university", "universidad", "kampus", "campus"},
    },
    "environment": {
        "environment": {"environment", "environmental", "lingkungan", "lingkungan hidup", "medio ambiente",
                        "meio ambiente", "ambiente", "environnement", "umwelt", "środowisko", "ekologia", "çevre",
                        "экология", "环境", "环保", "البيئة", "بيئة", "محیط زیست", "पर्यावरण", "환경", "môi trường",
                        "สิ่งแวดล้อม", "περιβάλλον"},
        "climate change": {"climate", "climate change", "climate crisis", "cambio climático", "klima", "iklim"},
        "nature": {"nature", "wildlife", "animals"},
        "sustainability": {"sustainability", "keberlanjutan"},
    },
    "health": {
        "health": {"health", "health news", "kesehatan", "sehat", "kesihatan", "salud", "saúde", "santé", "sante",
                   "gesundheit", "salute", "sağlık", "zdrowie", "здоровье", "здоров'я", "健康", "صحة", "الصحة", "سلامت",
                   "स्वास्थ्य", "सेहत", "건강", "sức khỏe", "สุขภาพ", "υγεία", "υγεια", "healthcare", "medis", "afya"},
        "mental health and disorder": {"mental health"},
        "communicable disease": {"coronavirus", "covid-19", "covid"},
        "medical profession": {"medicine", "medical", "medicina"},
    },
    "human interest": {
        "celebrity": {"celebrity", "celebrities", "celebs", "celebrity news", "seleb", "selebriti", "selebritas",
                      "gosip", "famosos", "celebridades", "célébrités", "ünlüler", "звёзды", "звезды", "明星", "مشاهير",
                      "सेलिब्रिटी", "셀럽"},
        "royalty": {"royals", "royal family", "the royals"},
        "human mishap": {"odd news", "offbeat", "weird news", "bizarre"},
    },
    "labour": {
        "labour": {"labour", "labor", "ketenagakerjaan", "tenaga kerja", "buruh", "lavoro", "arbeit", "praca", "труд",
                   "劳动", "노동", "çalışma hayatı"},
        "employment": {"employment", "empleo", "emprego", "emploi", "istihdam", "就业", "توظيف", "रोज़गार", "việc làm",
                       "careers", "career", "karier", "karir", "work & careers", "работа и карьера"},
        "labour market": {"arbeitsmarkt", "rynek pracy"},
        "unions": {"unions"},
    },
    "lifestyle and leisure": {
        "travel and tourism": {"travel", "travel news", "traveling", "wisata", "viajes", "viagem", "viaggi", "voyage",
                               "voyages", "reisen", "reise", "podróże", "seyahat", "путешествия", "旅行", "سفر",
                               "यात्रा", "여행", "du lịch", "ท่องเที่ยว"},
        "food and drink enthusiasm": {"food", "food & drink", "food and drink", "kuliner", "resep", "recipes",
                                      "recetas", "receitas", "ricette", "gastronomía", "gastronomia", "gastronomie",
                                      "yemek", "tarifler", "кулинария", "рецепты", "美食", "طبخ", "व्यंजन", "요리",
                                      "ẩm thực", "restaurants", "dining", "wine", "cuisine"},
        "house and home": {"home & garden", "home and garden", "garden", "gardening"},
        "hobby": {"hobbies"},
        "leisure": {"leisure"},
        "outdoor recreational activities": {"outdoors", "hunting", "fishing"},
        "exercise and fitness": {"fitness"},
        "wellness": {"wellness", "wellbeing", "well-being"},
        "gaming and lottery": {"lottery"},
    },
    "politics": {
        "politics and government": {
            "politics", "political", "politik", "politika", "política", "politica", "politique", "polityka",
            "politiek", "politiikka", "siyaset", "политика", "政治", "سياسة", "السياسة", "سیاسی", "राजनीति", "정치",
            "chính trị", "การเมือง", "πολιτική", "πολιτικη", "politica nazionale", "politica interna",
            "politik dalam negeri", "polityka krajowa", "політика", "політика україни", "ፖለቲካ"},
        "election": {"elections", "election", "election 2024", "pemilu", "pilkada", "elecciones", "eleições",
                     "élections", "wahlen", "seçim", "выборы"},
        "government": {"government", "pemerintahan", "gobierno", "parlamento", "parliament", "congress", "white house"},
        "diplomacy": {"diplomacy"},
    },
    "religion": {
        "religion": {"religion", "religions", "religión", "religione", "religião", "réligion", "religia", "agama",
                     "религия", "宗教", "دين", "الدين", "مذهب", "धर्म", "종교", "tôn giáo", "ศาสนา", "faith",
                     "faith & values", "spirituality", "religious"},
        "Islam": {"islam", "islami", "islamic", "khazanah islam"},
        "church": {"church", "iglesia", "chiesa"},
        "pope": {"vatican", "vaticano"},
    },
    "science and technology": {
        "science and technology": {
            "science", "science & technology", "science and technology", "sci-tech", "scitech", "sains",
            "sains dan teknologi", "iptek", "ciencia", "ciência", "scienza", "sciences", "wissenschaft", "nauka",
            "наука", "наука и техника", "наука и технологии", "科学", "علوم", "علم و فناوری", "विज्ञान", "과학",
            "khoa học", "bilim", "bilim teknoloji", "bilim ve teknoloji", "bilim-teknoloji", "επιστήμη",
            "technologie & wissenschaft"},
        "artificial intelligence": {"artificial intelligence", "ai"},
        "space exploration": {"space"},
    },
    "society": {
        "society": {"social issues", "social affairs"},
        "family": {"family", "parenting", "keluarga", "parenting & family", "familia", "família", "famiglia", "famille",
                   "familie", "rodzina", "aile", "семья", "家庭", "الأسرة", "परिवार", "가족", "gia đình"},
        "Dating and Relationships": {"relationships", "relationship", "dating"},
        "immigration": {"immigration", "migration", "inmigración", "migrazione", "göç", "миграция"},
        "demographic group": {"lgbt", "lgbtq", "lgbtq+", "gender", "gender equality"},
        "social condition": {"poverty", "kemiskinan"},
        "welfare": {"charity", "welfare"},
        "fundamental rights": {"human rights", "derechos humanos", "hak asasi manusia"},
        "demographics": {"demography", "population"},
    },
    "sport": {
        "sport": {"sport", "sports", "sport news", "sports news", "olahraga", "sukan", "deportes", "deporte",
                  "esportes", "desporto", "sportif", "sportivo", "spor", "спорт", "спорт новости", "体育", "體育",
                  "体育新闻", "الرياضة", "رياضة", "رياضه", "खेल", "스포츠", "スポーツ", "thể thao", "กีฬา", "sporten",
                  "sportul", "šport", "sportas", "urheilu", "idrett", "sporty", "ספורט", "ورزش", "ورزشی", "αθλητικά",
                  "αθλητισμός", "altri sport", "sport locali", "college sports", "high school sports", "prep sports",
                  "اخبار الرياضه", "أخبار الرياضة", "motori e sport",
                  # names used by the external datasets (L3Cube Hindi, Amharic press.et, Swahili news)
                  "khel", "ስፖርት", "michezo"},
        "competition discipline": {
            "football", "soccer", "calcio", "calciomercato", "fútbol", "futbol", "futebol", "fussball", "fußball",
            "sepak bola", "sepakbola", "bola", "liga 1", "liga inggris", "liga champions", "premier league", "la liga",
            "laliga", "serie a", "bundesliga", "ligue 1", "champions league", "europa league", "süper lig",
            "futbol haberleri", "футбол", "ποδόσφαιρο", "ποδοσφαιρο", "كرة القدم", "piłka nożna", "college football",
            "basketball", "basket", "baloncesto", "basquete", "баскетбол", "μπάσκετ", "nba", "nfl", "mlb", "nhl",
            "hockey", "хоккей", "tennis", "tenis", "теннис", "golf", "cricket", "ipl", "t20 world cup", "rugby",
            "nrl", "afl", "volley", "volleyball", "voli", "pallavolo", "voleibol", "formula 1", "formula one", "f1",
            "formula1", "motogp", "motorsport", "motorsports", "boxing", "boxeo", "mma", "ufc", "athletics",
            "atletismo", "cycling", "ciclismo", "badminton", "bulu tangkis", "wrestling", "baseball", "béisbol"},
        "Olympic Games": {"olympics", "olympic games", "juegos olímpicos", "olimpiadi"},
    },
    "weather": {
        "weather": {"weather", "weather news", "cuaca", "cuaca hari ini", "pogoda", "wetter", "météo", "meteo",
                    "meteorología", "meteorologia", "hava durumu", "погода", "天气", "天氣", "الطقس", "طقس", "آب و هوا",
                    "मौसम", "날씨", "thời tiết", "καιρός", "καιρος"},
        "weather forecast": {"weather forecast", "prakiraan cuaca", "previsioni meteo", "พยากรณ์อากาศ"},
        "weather phenomena": {"cuaca ekstrem"},
    },
}

# Section names that IPTC places under two top-level topics: never mapped (documented for the paper).
AMBIGUOUS: dict[str, str] = {
    **{t: "science and technology > technology and engineering  |  economy > products and services > "
          "computing and information technology"
       for t in ["tech", "tech news", "technology", "teknologi", "tekno", "gadget", "tecnología", "tecnologia",
                 "technologie", "technik", "технологии", "科技", "تكنولوجيا", "التكنولوجيا", "تقنية", "प्रौद्योगिकी",
                 "टेक", "công nghệ", "เทคโนโลยี", "teknoloji", "τεχνολογία", "internet", "mobile", "smartphones",
                 "innovation", "inovasi"]},
    "economics": "science and technology > social sciences > economics  |  economy, business and finance",
    **{t: "conflict, war and peace > act of terror  |  crime, law and justice > crime > terrorism"
       for t in ["terrorism", "terrorisme", "terrorismo", "terörle mücadele"]},
    **{t: "lifestyle and leisure > travel and tourism  |  economy > products and services > tourism and leisure industry"
       for t in ["tourism", "pariwisata", "turizm", "туризм", "旅游", "سياحة", "turismo"]},
    **{t: "human interest > ceremony > funeral and memorial service  |  society > values > death and dying"
       for t in ["obituaries", "obituary", "obits", "nekrolog"]},
    "energy & environment": "economy > products and services > energy and resource  |  environment",
    **{t: "arts > arts and entertainment  |  human interest > celebrity"
       for t in ["entertainment", "espectáculos", "cultura y espectáculos", "spettacoli", "hiburan", "연예", "娱乐",
                 "मनोरंजन", "show"]},
    **{t: "lifestyle and leisure  |  arts > fashion  |  health (catch-all lifestyle section)"
       for t in ["lifestyle", "gaya hidup", "estilo de vida", "vida y estilo", "yaşam", "stile di vita"]},
    "streaming": "arts > television / series  |  economy > media and entertainment industry",
    "market": "economy > market and exchange  |  classified / marketplace pages",
    "military": "conflict, war and peace > armed conflict  |  politics > national security > armed forces",
    "cyber": "conflict, war and peace > cyber warfare  |  crime, law and justice > crime > cyber crime",
}

# Terms whose meaning depends on the article language (None = do not map in that language)
LANG_TERMS: dict[str, dict[str, str | None]] = {
    "tr": {"magazin": "human interest", "din": "religion", "ekonomi": "economy, business and finance",
           "yaşam": None},
    "id": {"otomotif": None},
    # Hindi religion sections are dominated by astrology calendars and "auspicious day" tips
    "hi": {"धर्म": None, "आस्था": None, "धर्म कर्म": None, "religion": None, "faith": None,
           "spirituality": None, "spiritual": None, "अध्यात्म": None},
    "de": {"panorama": None},
}

# An article carrying any of these names is rejected whatever else it maps to: astrology and
# horoscopes have no IPTC Media Topic and often share a section with religion.
REJECT_TERMS = {
    "राशिफल", "ज्योतिष", "अंक ज्योतिष", "वास्तु", "horoscope", "horoscopes", "horóscopo", "horoscopo", "oroscopo",
    "horoskop", "burç", "burçlar", "гороскоп", "гороскопы", "astrology", "astrología", "astrologia", "tarot",
    "zodiac", "zodiak", "ramalan", "星座", "占星", "الأبراج", "abraj", "astro",
}

_GENERIC = (r"(latest |top |breaking |berita |noticias de |news |haberleri$|haber$|news$|berita$| news| nachrichten"
            r"| новости|^новости |^اخبار |^أخبار )")


def normalise(section: str) -> str:
    s = unicodedata.normalize("NFKC", section).lower().strip()
    s = s.replace("&amp;", "&").replace("’", "'")
    s = re.sub(r"[\"“”«»()\[\]|/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—:·.,")
    return s.replace("̇", "")  # Turkish "İ".lower() leaves a combining dot: asayi̇ş -> asayiş


def _validate() -> None:
    """Check every mapped IPTC node against the official tree and that no term is mapped twice."""
    assert set(MAPPING) == set(LABELS), "MAPPING must cover exactly the 17 labels"
    seen: dict[str, str] = {}
    for label, groups in MAPPING.items():
        for terms in groups.values():
            for t in terms:
                assert t == normalise(t), f"term not normalised: {t!r}"
                assert t not in AMBIGUOUS, f"ambiguous term is mapped: {t!r}"
                assert seen.setdefault(t, label) == label, f"{t!r} mapped to {seen[t]} and {label}"
    if not REFERENCE.exists():
        return
    nodes = [n for n in json.loads(REFERENCE.read_text(encoding="utf8")) if not n["retired"]]
    by_name = {}
    for n in nodes:
        by_name.setdefault(n["name"], []).append(n)
    for label, groups in MAPPING.items():
        for node in groups:
            cands = by_name.get(node)
            assert cands, f"IPTC node not found (or retired): {node!r}"
            assert any(c["top_name"] == IPTC_TOP[label] for c in cands), \
                f"IPTC node {node!r} is not under {IPTC_TOP[label]!r}: {[c['top_name'] for c in cands]}"


_validate()
_LOOKUP = {t: label for label, groups in MAPPING.items() for terms in groups.values() for t in terms}


def map_section(section: str, lang: str | None = None) -> str | None:
    """IPTC label for one section name, or None when the name is not unambiguously mapped."""
    s = normalise(section)
    if not s or len(s) > 40:  # long strings are headlines / sentences, not section names
        return None
    if lang in LANG_TERMS and s in LANG_TERMS[lang]:
        return LANG_TERMS[lang][s]
    if s in _LOOKUP:
        return _LOOKUP[s]
    stripped = re.sub(_GENERIC, " ", s).strip()  # "sports news", "berita olahraga" -> "sports"
    if stripped and stripped != s:
        return _LOOKUP.get(stripped)
    return None


def label_article(sections: list[str], breadcrumb: list[str], lang: str | None) -> tuple[str | None, str]:
    """Label an article from its articleSection values (preferred) and breadcrumb names.

    Returns (label, reason). The label is set only if every mapped name agrees on one IPTC topic;
    any conflict (e.g. "Sport" and "Crime") rejects the article.
    """
    if any(normalise(s) in REJECT_TERMS for s in sections + breadcrumb):
        return None, "rejected_term"
    from_section = {map_section(s, lang) for s in sections} - {None}
    from_crumbs = {map_section(s, lang) for s in breadcrumb[1:]} - {None}  # crumb 0 = home page
    mapped = from_section | from_crumbs
    if not mapped:
        return None, "unmapped"
    if len(mapped) > 1:
        return None, "conflict"
    label = next(iter(mapped))
    return label, "section" if from_section else "breadcrumb"


def export_table(path: Path) -> None:
    """Mapping table for the paper: label, IPTC node id and path, section names."""
    import csv
    nodes = {n["id"]: n for n in json.loads(REFERENCE.read_text(encoding="utf8"))}
    by_name = {}
    for n in nodes.values():
        if not n["retired"]:
            by_name.setdefault(n["name"], []).append(n)

    def node_path(mid):
        names = []
        while mid:
            names.append(nodes[mid]["name"])
            mid = nodes[mid]["parent"]
        return " > ".join(reversed(names))

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "iptc_node_id", "iptc_node_path", "n_terms", "section_names"])
        for label, groups in MAPPING.items():
            for node, terms in groups.items():
                n = next(c for c in by_name[node] if c["top_name"] == IPTC_TOP[label])
                w.writerow([label, n["id"], node_path(n["id"]), len(terms), "; ".join(sorted(terms))])
        for t, why in sorted(AMBIGUOUS.items()):
            w.writerow(["(not mapped: ambiguous)", "", why, 1, t])


if __name__ == "__main__":
    out = REFERENCE.with_name("mapping_table.csv")
    export_table(out)
    n_terms = sum(len(t) for g in MAPPING.values() for t in g.values())
    print(f"validated {sum(len(g) for g in MAPPING.values())} IPTC nodes, {n_terms} section names, "
          f"{len(AMBIGUOUS)} ambiguous names -> {out}")
