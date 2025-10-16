# Imports
from itertools import combinations
from pyspark import SparkContext
from typing import Tuple

from pyspark.sql import (
    SparkSession,
    DataFrame,
    functions as f
)
from pyspark.sql.functions import (
    collect_list,
    explode,
    col,
    size,
)
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
)

class _PCYParams:
    def __init__(
            self,
            min_support: int = 100,
            min_confidence: float = 0.8,
            num_buckets: int = 5000,
            item_col: str = "item_list"
    ) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.num_buckets = num_buckets
        self.item_col = item_col
    # --- end __init__

    # getters
    def get_min_support(self) -> int:
        return self.min_support
    def get_min_confidence(self) -> float:
        return self.min_confidence
    def get_num_buckets(self) -> int:
        return self.num_buckets
    def get_item_col(self) -> str:
        return self.item_col
    # --- end getters

    # setters
    def set_min_support(self, min_support: int):
        self.min_support = min_support
        return self
    def set_min_confidence(self, min_confidence: float):
        self.min_confidence = min_confidence
        return self
    def set_num_buckets(self, num_buckets: int):
        self.num_buckets = num_buckets
        return self
    def set_item_col(self, item_col: str):
        self.item_col = item_col
        return self
    # --- end setters
# -- end class _PCYParams

class PCYModel(_PCYParams):
    def __init__(
            self,
            min_support: int = 100,
            min_confidence: float = 0.8,
            num_buckets: int = 5000,
            frequent_pairs: DataFrame = None,
            association_rules: DataFrame = None,
            item_col: str = "item_list"
    ) -> None:
        super().__init__(min_support, min_confidence, num_buckets, item_col)
        self.frequent_pairs = frequent_pairs
        self.association_rules = association_rules
    # -- end __init__

    def show_frequent_pairs(self, n: int = 5, truncate: bool = False):
        print(f"Top {n} frequent pairs found:")
        self.frequent_pairs.show(n, truncate=truncate)
    # -- end show_frequent_pairs

    def show_association_rules(self, n: int = 10, truncate: bool = False):
        print(f"Top {n} association rules found (ordered by confidence):")
        self.association_rules.orderBy(col("conf").desc()).show(n, truncate=truncate)
    # -- end show_association_rules
# -- end class PCYModel

class PCY(_PCYParams):
    # Tìm tập phổ biến I -> phổ biến ? -> s
    # s,c,i: constant
    def __init__(
            self,
            min_support: int = 100,
            min_confidence: float = 0.8,
            num_buckets: int = 5000,
            item_col: str = "item_list"
    ) -> None:
        super().__init__(min_support, min_confidence, num_buckets, item_col)
    # -- end __init__

    def _pass1(self, baskets: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        # Pass 1:
        # FOR (each basket) :
        #     FOR (each item in the basket):
        #         add 1 to item's count;
        item_cols_name = self.get_item_col()

        item_counts = baskets.select(f.explode(item_cols_name).alias("item")) \
                             .groupBy("item") \
                             .count()
        item_counts.cache()
        
        frequent_items = item_counts.filter(f.col("count") >= self.get_min_support()) \
                                    .select("item")
        frequent_items.cache()
        
        baskets_with_id = baskets.withColumn("basket_id", f.monotonically_increasing_id())
        exploded_baskets = baskets_with_id.select("basket_id", f.explode(item_cols_name).alias("item"))

        # If item i does not appear in s baskets, 
        # then no pair including i can appear in s baskets
        frequent_items_in_baskets = exploded_baskets.join(frequent_items, "item", "inner")
        filtered_baskets = frequent_items_in_baskets.groupBy("basket_id") \
                                                    .agg(f.collect_list("item").alias("freq_items"))

        #     FOR (each pair of items in the basket):
        #         hash the pair to a bucket;
        #         add 1 to the count for that bucket.
        
        # def hash_pair(items: list) -> list[int]:
        #     items = sorted(items)
        #     return [f.hash(pair) % self.get_num_buckets() for pair in combinations(items, 2)]

        def hash_pair(items: list) -> list[int]:
            items = sorted(items)
            return [
                hash(f"{pair[0]}|{pair[1]}") % self.get_num_buckets() 
                for pair in combinations(items, 2)
            ]
            
        hash_udf = f.udf(hash_pair, ArrayType(IntegerType()))

        bucket_counts = filtered_baskets.withColumn("buckets", hash_udf(f.col("freq_items"))) \
                                        .select(f.explode("buckets").alias("bucket_index")) \
                                        .groupBy("bucket_index") \
                                        .count()

        frequent_buckets = bucket_counts.filter(f.col("count") >= self.get_min_support())
        frequent_buckets.cache()

        return item_counts, frequent_items, frequent_buckets
    # -- end _pass1
    
    def _pass2(self, baskets: DataFrame, frequent_items: DataFrame, frequent_buckets: DataFrame) -> DataFrame:
        # Pass 2:
        def pair_gen(items: list[str]) -> list[tuple[str, str]]:
            return list(combinations(sorted(items), 2))
        
        item_cols_name = self.get_item_col()

        pair_gen_udf = f.udf(pair_gen, ArrayType(ArrayType(StringType())))
        all_pairs = baskets.withColumn("pairs", pair_gen_udf(f.col(item_cols_name))) \
                           .select(f.explode("pairs").alias("pair")) \
                           .select(f.col("pair")[0].alias("item1"), f.col("pair")[1].alias("item2"))
        
        freq_item_to_join = frequent_items.select(f.col("item").alias("item_alias"))

        candidates = all_pairs.join(freq_item_to_join, all_pairs.item1 == freq_item_to_join.item_alias, "inner") \
                             .select("item1", "item2")
        candidates = candidates.join(freq_item_to_join, candidates.item2 == freq_item_to_join.item_alias, "inner") \
                             .select("item1", "item2")
        # Cache candidates as it's used for bucket joining
        candidates.cache()
        
        # pair_with_bucket = candidates.withColumn(
        #     "bucket_index",
        #     (f.hash(f.col("item1"), f.col("item2")) % self.get_num_buckets())
        # )

        def hash_single_pair(item1: str, item2: str) -> int:
            items = sorted([item1, item2])
            return hash(f"{items[0]}|{items[1]}") % self.get_num_buckets()

        hash_single_udf = f.udf(hash_single_pair, IntegerType())
        pair_with_bucket = candidates.withColumn(
            "bucket_index",
            hash_single_udf(f.col("item1"), f.col("item2"))
        )
        
        filtered_pairs = pair_with_bucket.join(
            frequent_buckets.select("bucket_index"),
            "bucket_index",
            "inner"
        ).select("item1", "item2")

        frequent_pairs = filtered_pairs.groupBy("item1", "item2").count() \
                                    .filter(f.col("count") >= self.get_min_support()) \
                                    .select("item1", "item2", f.col("count").alias("support")) \
                                    .orderBy(f.col("support").desc())
        # Cache frequent_pairs as it's used for rule generation
        frequent_pairs.cache()
        
        return frequent_pairs
    # -- end _pass2
    
    def _generate_rules(self, frequent_pairs: DataFrame, item_counts: DataFrame) -> DataFrame:
        # A -> B
        a_b = frequent_pairs.join(item_counts, frequent_pairs.item1 == item_counts.item)
        a_b = a_b.withColumn("conf", f.col("support") / f.col("count")).filter(f.col("conf") >= self.get_min_confidence())
        a_b = a_b.select("item1", "item2", "support", "conf")

        # B -> A:
        b_a = frequent_pairs.join(item_counts, frequent_pairs.item2 == item_counts.item)
        b_a = b_a.withColumn("conf", f.col("support") / f.col("count")).filter(f.col("conf") >= self.get_min_confidence())
        
        b_a = b_a.select(
            f.col("item2").alias("item1"), 
            f.col("item1").alias("item2"), 
            "support", 
            "conf"
        )

        association_rules = a_b.union(b_a)
        association_rules.cache()
        
        return association_rules
    # -- end _generate_rules
    
    def fit(self, baskets: DataFrame) -> PCYModel:
        # Pass 1
        item_counts, frequent_items, frequent_buckets = self._pass1(baskets)
        # Pass 2
        frequent_pairs = self._pass2(baskets, frequent_items, frequent_buckets)

        association_rules = self._generate_rules(frequent_pairs, item_counts)

        return PCYModel(
            min_support = self.get_min_support(),
            min_confidence = self.get_min_confidence(),
            num_buckets = self.get_num_buckets(),
            item_col = self.get_item_col(),
            frequent_pairs = frequent_pairs,
            association_rules = association_rules
        )
# -- end class PCY