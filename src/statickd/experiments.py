"""The registry of every StaticKD training run reported in the paper.

Each run changes ONE factor with respect to the final recipe (StaticKD, one seed):
    potion-multilingual init, fine-tuned embeddings, linear head, soft teacher labels with tau = 1,
    token dropout 0.2, 285,999 distillation documents in 69 languages, 6 epochs.
The released StaticKD is the table average of the seeds 0, 1 and 2 of the final recipe (fixed before
the unified test set was built). All runs are trained with the same code (statickd.student.train).
"""
from statickd.student import StaticKDConfig

MAIN_SEEDS = tuple(range(10))       # final recipe: 10 seeds (variance + ensemble-size analysis)
ABLATION_SEEDS = tuple(range(5))    # every ablation / temperature setting: 5 seeds
EXPLORATION_SEEDS = tuple(range(3))  # negative-result variants: 3 seeds
V2_MEMBERS = (0, 1, 2)              # seeds averaged into the released StaticKD

# code -> (description, config overrides, seeds)
ABLATIONS = {
    "full":         ("Resep final (satu seed)", {}, MAIN_SEEDS),
    "hard":         ("Label keras teacher (arg-max, cross-entropy) alih-alih soft label",
                     {"loss": "hard"}, ABLATION_SEEDS),
    "emm_only":     ("Data distilasi hanya EMMediaTopic (4 bahasa, 20 rb dokumen)",
                     {"sources": ("emmediatopic",), "epochs": 10}, ABLATION_SEEDS),
    "data130k":     ("Data distilasi 130 rb dokumen (tanpa CC-News tambahan)",
                     {"sources": ("emmediatopic", "ccnews")}, ABLATION_SEEDS),
    "random_init":  ("Embedding diinisialisasi acak (tanpa potion-multilingual)",
                     {"init": "random"}, ABLATION_SEEDS),
    "frozen":       ("Embedding potion dibekukan, hanya kepala linear dilatih",
                     {"freeze_embeddings": True}, ABLATION_SEEDS),
    "no_tdrop":     ("Tanpa token dropout", {"token_dropout": 0.0}, ABLATION_SEEDS),
    "mlp":          ("Kepala MLP 512 unit (tidak dapat dilipat)", {"hidden": 512}, ABLATION_SEEDS),
    "tau0.5":       ("Temperatur tau = 0,5", {"temperature": 0.5}, ABLATION_SEEDS),
    "tau2":         ("Temperatur tau = 2", {"temperature": 2.0}, ABLATION_SEEDS),
    "tau4":         ("Temperatur tau = 4", {"temperature": 4.0}, ABLATION_SEEDS),
    "bigram":       ("+ tabel bigram hashing 2^21", {"bigram_bits": 21}, EXPLORATION_SEEDS),
    "idf":          ("+ pooling berbobot idf^0,5", {"idf_power": 0.5}, EXPLORATION_SEEDS),
    "conf":         ("+ bobot dokumen = keyakinan teacher", {"conf_weight": 1.0}, EXPLORATION_SEEDS),
    "gpt_ce":       ("+ cross-entropy label GPT-4o (bobot 1)", {"gpt_weight": 1.0}, EXPLORATION_SEEDS),
}


def run_name(code: str, seed: int) -> str:
    return f"{code}_s{seed}"


def config_for(code: str, seed: int) -> StaticKDConfig:
    _, overrides, _ = ABLATIONS[code]
    return StaticKDConfig(name=run_name(code, seed), seed=seed, **overrides)


def all_runs() -> list[tuple[str, int]]:
    return [(code, seed) for code, (_, _, seeds) in ABLATIONS.items() for seed in seeds]
