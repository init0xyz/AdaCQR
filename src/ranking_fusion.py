import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense_candidates_path', type=str)
    parser.add_argument('--sparse_candidates_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    with open(args.dense_candidates_path, "r") as f1, open(args.sparse_candidates_path, "r") as f2, open(args.output_path, "w") as f_out, open(args.data_path, "r") as f_data:
        for line1, line2, line3 in zip(f1, f2, f_data):
            data = json.loads(line3)
            sample_id = data["sample_id"]
            dense_candidates = json.loads(line1)
            sparse_candidates = json.loads(line2)
            assert len(dense_candidates) == len(sparse_candidates)
            assert sample_id == dense_candidates["sample_id"]
            assert sample_id == sparse_candidates["sample_id"]

            candidates = []

            for sparse_cand, dense_cand in zip(sparse_candidates["candidates"], dense_candidates["candidates"]):
                assert sparse_cand[0] == dense_cand[0]
                cur_score = sparse_cand[1] + dense_cand[1]
                candidates.append([sparse_cand[0], cur_score])
            
            data["candidates"] = candidates
            f_out.write(json.dumps(data) + "\n")
    print("Done!")