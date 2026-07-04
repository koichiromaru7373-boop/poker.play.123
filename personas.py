"""AIペルソナ定義: トーナメントの席1〜5に座るキャラクター

スタイル4類型(LAG/TAG/ルースパッシブ/タイトパッシブ)+学習型AIの5人。
名前・性格説明はAPIがそのまま返すので、サイト側で表示して「人読み」の
練習材料にする。
"""

from my_agents import StyleAgent

PERSONAS = [
    {
        "key": "lag",
        "name": "嵐山レン",
        "style": "ルース・アグレッシブ (LAG)",
        "description": "ほぼ全ハンドに参加してガンガンレイズしてくる暴れ馬。"
                       "ブラフも多いので、強い手でじっくり受け止めると大きく取れる。",
        "params": {"looseness": 0.9, "aggression": 0.85},
    },
    {
        "key": "tag",
        "name": "白鷺ミサキ",
        "style": "タイト・アグレッシブ (TAG)",
        "description": "選び抜いた強いハンドだけで参加し、入ったら積極的に攻める教科書型。"
                       "彼女のレイズは本物が多い。安易にコールしないこと。",
        "params": {"looseness": 0.25, "aggression": 0.8},
    },
    {
        "key": "lp",
        "name": "ゴンさん",
        "style": "ルース・パッシブ (コーリングステーション)",
        "description": "とにかくコールで付いてくるおじさん。ブラフはほぼ通じない。"
                       "強い手ができたら大きくベットすれば最後まで払ってくれる。",
        "params": {"looseness": 0.85, "aggression": 0.15},
    },
    {
        "key": "tp",
        "name": "イワオ",
        "style": "タイト・パッシブ (ロック)",
        "description": "滅多に参加せず、攻めることも少ない超堅実派。"
                       "彼がレイズしたときはモンスター級。すぐ降りるのが正解。",
        "params": {"looseness": 0.15, "aggression": 0.2},
    },
    {
        "key": "champion",
        "name": "ゼロ",
        "style": "学習型AI (スタイル不明)",
        "description": "自己対戦で鍛えられた学習型AI。決まった型がなく読みにくい、"
                       "このテーブルの最強格。",
        "params": None,  # None = 学習済みモデルを使う
    },
]


def build_tournament_agents(load_model_fn, num_actions=5):
    """席1〜5のエージェント辞書と席情報辞書を返す

    load_model_fn: 学習済みモデルを返す関数(championに使う)
    """
    agents, seats_info = {}, {}
    for i, p in enumerate(PERSONAS, start=1):
        if p["params"] is None:
            agents[i] = load_model_fn()
        else:
            agents[i] = StyleAgent(num_actions=num_actions, seed=i, **p["params"])
        seats_info[i] = {
            "name": p["name"],
            "style": p["style"],
            "description": p["description"],
        }
    return agents, seats_info
