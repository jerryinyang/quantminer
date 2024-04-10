from typing import List, Union

import numpy as np


class PIP:
    def __init__(self, n_pivots: int, dist_measure: int) -> None:
        """
        Arguments:
        - n_pivots : Number of pivots
        - dist_measure
            * 1 = Euclidean Distance
            * 2 = Perpindicular Distance
            * 3 = Vertical Distance
        """
        self.n_pivots = n_pivots
        self.dist_measure = dist_measure


    def transform(self, data: Union[np.ndarray, List[np.ndarray]]):
        data = np.array(data)

        if data.ndim > 1:
            pips = []

            for _data in data:
                pips.append(self.transform(_data))

            return np.array(pips)

        n_pivots = self.n_pivots
        dist_measure = self.dist_measure

        pips_indices = [0, len(data) - 1]  # Index
        pips_prices = [data[0], data[-1]]  # Price

        for curr_point in range(2, n_pivots):
            md = 0.0  # Max distance
            md_i = -1  # Max distance index
            insert_index = -1

            for k in range(0, curr_point - 1):
                # Left adjacent, right adjacent indices
                left_adj = k
                right_adj = k + 1

                time_diff = pips_indices[right_adj] - pips_indices[left_adj]
                price_diff = pips_prices[right_adj] - pips_prices[left_adj] + 1e-15
                slope = price_diff / time_diff

                intercept = pips_prices[left_adj] - pips_indices[left_adj] * slope

                for i in range(pips_indices[left_adj] + 1, pips_indices[right_adj]):
                    d = 0.0  # Distance
                    if dist_measure == 1:  # Euclidean distance
                        d = (
                            (pips_indices[left_adj] - i) ** 2
                            + (pips_prices[left_adj] - data[i]) ** 2
                        ) ** 0.5
                        d += (
                            (pips_indices[right_adj] - i) ** 2
                            + (pips_prices[right_adj] - data[i]) ** 2
                        ) ** 0.5
                    elif dist_measure == 2:  # Perpindicular distance
                        d = (
                            abs((slope * i + intercept) - data[i])
                            / (slope**2 + 1) ** 0.5
                        )
                    else:  # Vertical distance
                        d = abs((slope * i + intercept) - data[i])

                    if d > md:
                        md = d
                        md_i = i
                        insert_index = right_adj

            pips_indices.insert(insert_index, md_i)
            pips_prices.insert(insert_index, data[md_i])

        return np.array(pips_prices)
