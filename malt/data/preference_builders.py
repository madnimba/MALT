from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class GeneratorSftSample:
    question: str
    generator_output: str


@dataclass
class VerifierSftSample:
    question: str
    generator_output: str
    verifier_output: str


@dataclass
class RefinerSftSample:
    question: str
    generator_output: str
    verifier_output: str
    refiner_output: str


@dataclass
class VerifierDpoSample:
    question: str
    generator_output: str
    chosen: str
    rejected: str


@dataclass
class RefinerDpoSample:
    question: str
    generator_output: str
    verifier_output: str
    chosen: str
    rejected: str


def build_generator_sft_samples(
    trajectories: Iterable[Dict],
) -> List[GeneratorSftSample]:
    """
    Build SFT training samples for the Generator from valued trajectories.

    Uses only generator nodes with label == 1.
    """
    samples: List[GeneratorSftSample] = []
    for traj in trajectories:
        q = traj.get("question", "")
        for g in traj.get("generator_nodes", []):
            if int(g.get("label", 0)) != 1:
                continue
            samples.append(
                GeneratorSftSample(
                    question=q,
                    generator_output=g.get("text", ""),
                )
            )
    return samples


def build_verifier_and_refiner_sft_samples(
    trajectories: Iterable[Dict],
) -> Tuple[List[VerifierSftSample], List[RefinerSftSample]]:
    """
    Build SFT training samples for Verifier and Refiner.

    - Verifier SFT uses verifier nodes with label == 1.
    - Refiner SFT uses refiner nodes with value == 1.
    """
    v_samples: List[VerifierSftSample] = []
    r_samples: List[RefinerSftSample] = []

    for traj in trajectories:
        q = traj.get("question", "")
        for g in traj.get("generator_nodes", []):
            g_text = g.get("text", "")
            for v in g.get("verifier_nodes", []):
                v_text = v.get("text", "")
                if int(v.get("label", 0)) == 1:
                    v_samples.append(
                        VerifierSftSample(
                            question=q,
                            generator_output=g_text,
                            verifier_output=v_text,
                        )
                    )
                for r in v.get("refiner_nodes", []):
                    if int(r.get("value", 0)) == 1:
                        r_samples.append(
                            RefinerSftSample(
                                question=q,
                                generator_output=g_text,
                                verifier_output=v_text,
                                refiner_output=r.get("text", ""),
                            )
                        )

    return v_samples, r_samples


def build_verifier_dpo_samples(
    trajectories: Iterable[Dict],
) -> List[VerifierDpoSample]:
    """
    Build preference pairs for Verifier DPO.

    For each generator node, we form (v_pos, v_neg) pairs where v_pos.label == 1
    and v_neg.label == 0.
    """
    pairs: List[VerifierDpoSample] = []

    for traj in trajectories:
        q = traj.get("question", "")
        for g in traj.get("generator_nodes", []):
            g_text = g.get("text", "")
            verifiers = g.get("verifier_nodes", [])
            positives = [v for v in verifiers if int(v.get("label", 0)) == 1]
            negatives = [v for v in verifiers if int(v.get("label", 0)) == 0]

            if not positives or not negatives:
                continue

            for v_pos in positives:
                for v_neg in negatives:
                    pairs.append(
                        VerifierDpoSample(
                            question=q,
                            generator_output=g_text,
                            chosen=v_pos.get("text", ""),
                            rejected=v_neg.get("text", ""),
                        )
                    )

    return pairs


def build_refiner_dpo_samples(
    trajectories: Iterable[Dict],
) -> List[RefinerDpoSample]:
    """
    Build preference pairs for Refiner DPO.

    For each (generator, verifier) pair, we form (r_pos, r_neg) pairs where
    r_pos.value == 1 and r_neg.value == 0.
    """
    pairs: List[RefinerDpoSample] = []

    for traj in trajectories:
        q = traj.get("question", "")
        for g in traj.get("generator_nodes", []):
            g_text = g.get("text", "")
            for v in g.get("verifier_nodes", []):
                v_text = v.get("text", "")
                refiners = v.get("refiner_nodes", [])
                positives = [r for r in refiners if int(r.get("value", 0)) == 1]
                negatives = [r for r in refiners if int(r.get("value", 0)) == 0]

                if not positives or not negatives:
                    continue

                for r_pos in positives:
                    for r_neg in negatives:
                        pairs.append(
                            RefinerDpoSample(
                                question=q,
                                generator_output=g_text,
                                verifier_output=v_text,
                                chosen=r_pos.get("text", ""),
                                rejected=r_neg.get("text", ""),
                            )
                        )

    return pairs

