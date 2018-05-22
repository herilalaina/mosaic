class RAVE():
    def __init__(self):
        self.rave_scores = {}

    def update(self, moves, score):
        len_moves = len(moves)

        if len_moves <= 1:
            # Nothing to do
            return

        source = moves[0][0] + "_" + str(moves[0][1])
        for i in range(1, len_moves):
            destination = moves[i][0] + "_" + str(moves[i][1])
            if source in self.rave_scores:
                if destination in self.rave_scores[source]:
                    info_action = self.rave_scores[source][destination]
                    new_value = info_action["v"] + (score - info_action["v"]) / (info_action["n"] + 1)
                    self.rave_scores[source][destination] = {"v": new_value, "n": info_action["n"] + 1}
                else:
                    self.rave_scores[source][destination] = {"v": score, "n": 1}
            else:
                self.rave_scores[source] = {}
                self.rave_scores[source][destination] = {"v": score, "n": 1}
            source = destination

    def get_score(self, sour, dest):
        try:
            return self.rave_scores[sour][dest]["v"]
        except Exception:
            return 0
