import argparse
import json
from tqdm import tqdm

import ipdb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

BELIEF_STATE = " => Belief State : "
EOB = " <EOB> "


def calculate_simmilarity(input_sent, cand_sent):
    # tokenization
    X_list = word_tokenize(input_sent)
    Y_list = word_tokenize(cand_sent)

    # sw contains the list of stopwords
    sw = stopwords.words("english")
    l1 = []
    l2 = []

    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / (float((sum(l1) * sum(l2)) ** 0.5) + 0.000001)
    return cosine


def postprocess_retrieval_devtest(
    lines, dialogue_id, domain, all_candidates, system_transcript_pool
):
    output = {"dialog_id": dialog_id, "predictions": []}
    scores = {"dialog_id": dialog_id, "candidate_scores": []}

    turn_id = 0
    for line in lines:
        prediction = {"response": "", "turn_id": turn_id}
        retrieval_score = {"scores": [], "turn_id": turn_id}

        candidates = all_candidates[turn_id]["retrieval_candidates"]
        response = ""
        parse = line.split(BELIEF_STATE)[1]
        generated = parse.split(EOB)
        if EOB in line:
            response = generated[1].lstrip().strip()
        for candidate in candidates:
            system_utt = system_transcript_pool[domain][candidate]
            retrieval_score["scores"].append(
                calculate_simmilarity(system_utt, response) * 100
            )
        prediction["response"] = response
        output["predictions"].append(prediction)
        scores["candidate_scores"].append(retrieval_score)
        turn_id += 1
    return output, scores


def postprocess_retrieval_teststd(
    lines, dialogue_id, domain, all_candidates, system_transcript_pool
):
    output = {"dialog_id": dialog_id, "predictions": []}
    scores = {"dialog_id": dialog_id, "candidate_scores": []}

    turn_id = 0
    for line in lines:
        prediction = {"response": "", "turn_id": turn_id}
        retrieval_score = {"scores": [], "turn_id": turn_id}

        response = ""
        parse = line.split(BELIEF_STATE)[1]
        generated = parse.split(EOB)
        if EOB in line:
            response = generated[1].lstrip().strip()

        prediction["response"] = response
        output["predictions"].append(prediction)

        if turn_id == int(all_candidates['turn_idx']):
            candidates = all_candidates['retrieval_candidates']
        #  candidates = all_candidates[turn_id]["retrieval_candidates"]
            for candidate in candidates:
                system_utt = system_transcript_pool[domain][candidate]
                retrieval_score["scores"].append(
                    calculate_simmilarity(system_utt, response) * 100
                )
            scores["candidate_scores"].append(retrieval_score)

        turn_id += 1
    return output, scores


def find_candidates(dialog_id, all_candidates):
    for candidate in all_candidates:
        if candidate["dialogue_idx"] == dialog_id:
            return candidate

    return None


if __name__ == "__main__":
    # Parse input args")
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_path", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--dials_path", type=str, required=True)
    parser.add_argument("--retrieval_candidate_path", type=str, required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    predicted = open(args.predicted_path, "r")
    predicted.seek(0)
    predicted_processed_retrieval = open(
        f"{args.output_path}/dstc10-simmc-{args.domain}-pred-subtask-4-retrieval.json", 'w'
    )
    predicted_processed_generation = open(
        f"{args.output_path}/dstc10-simmc-{args.domain}-pred-subtask-4-generation.json", 'w'
    )
    dials_path = args.dials_path
    retrieval_candidate = args.retrieval_candidate_path
    domain = args.domain

    idx = 0
    dialog_ids = []
    turn_length = []
    response_result = []
    retrieval_result = []
    candidates = []
    with open(args.retrieval_candidate_path, "r") as file:
        system_transcript_pool = json.load(file)["system_transcript_pool"]
    with open(args.retrieval_candidate_path, "r") as file:
        all_candidates = json.load(file)["retrieval_candidates"]

    with open(dials_path, "r") as file:
        dials = json.load(file)
        for dialogue in dials["dialogue_data"]:
            leng = len(dialogue["dialogue"])
            dialog_ids.append((dialogue["dialogue_idx"], dialogue["domain"]))
            turn_length.append(leng)

    i = 0
    with tqdm(total=len(dialog_ids)) as pbar:
        for dialog_id, domain in dialog_ids:
            lines = []
            candidates = find_candidates(dialog_id, all_candidates)["retrieval_candidates"]
            for t in range(turn_length[i]):
                try:
                    lines.append(next(predicted))
                except StopIteration:
                    pass
            if args.domain == 'devtest':
                result = postprocess_retrieval_devtest(
                    lines, dialog_id, domain, candidates, system_transcript_pool
                )
            elif args.domain == 'teststd':
                result = postprocess_retrieval_teststd(
                    lines, dialog_id, domain, candidates, system_transcript_pool
                )
            else:
                raise ValueError("Invalid domain")

            response_result.append(result[0])
            retrieval_result.append(result[1])
            i += 1
            pbar.update(1)
            # print("Finished converting dialog id : {}".format(dialog_id))

    json.dump(response_result, predicted_processed_generation)
    json.dump(retrieval_result, predicted_processed_retrieval)
    #  ipdb.set_trace(context=10)
    predicted_processed_generation.close()
    predicted_processed_retrieval.close()
    predicted.close()
    print(
        "Done converting {} total dialog to task 4 output format".format(i)
    )
