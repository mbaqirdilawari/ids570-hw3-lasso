**Homework 3: L1 vs. L2 logistic regression**

**Setup.** 

The updated Canvas files give 571 texts and 9,773 CORE chunks. Both models use the same train/test JSON (15,636 / 3,910 chunks) and the same TF-IDF settings (min_df=5, max_df=0.9, 16,228 features).

**Performance.** 

Measured against our labels, L1 is better. 
CORE F1 (the F1 score for the CORE class, label 1) rises from 0.938 (L2) to 0.998 (L1), and ROC AUC from 0.987 to 0.999. 
L2 made 77 false positives and 155 false negatives, while L1 made 0 and 8.

**Sparsity, and how the top words differ between L2 and L1.** 

L2 keeps all 16,228 coefficients non-zero, L1 keeps 30 (0.18%). L2's top positive words mix our seed terms (trade, merchants, merchant, factor, staple, chapman) with context words we never supplied (commodities, buy, exchange, london, goods, ships). Its negatives describe religion and war (god, christ, rome, pope, doctrine, armie). 
L1's top 12 positives are our Tier A/B keywords with huge weights (trade 175.8, merchants 148.1), then small weights on ship, pounds and filler such as "have", only 6 words are negative (god, that, it, christ, not, his). L1 is more readable only narrowly. The list is short, but it mostly says "contains one of our keywords," which we already knew.

**Choosing winners.** 

Yes. L2 spreads weight across ship (2.62), ships (2.74), shipping, goods, wares, buy, sell and exchange. L1 keeps ship (2.41) and sets ships, shipping, goods, buy, sell and exchange to exactly zero. It even zeroes two of our own keywords, venturer and staplers. The downside is that the winner among near-duplicates is arbitrary and could change with another sample, and a chunk using only a dropped variant loses that evidence. A zero does not mean a word is irrelevant.

**Label noise.** 

A plain "contains a Tier A/B word" check matches the test labels 100% of the time, and L1 agrees with that check 99.8% of the time (L2 agrees 94.1% of the time). I read 10 L2 errors (5 false positives and 5 false negatives, chosen at random) and all 8 L1 errors. For L2, I judged the label wrong in 6 of 10. The false positives were about commodities, prices and a salt scheme saving "twenty thousand pound," and the false negatives used "trade" as a metaphor ("common trade of living together"). L1 copied the label in all 6. For L1, 4 of 8 errors were label errors (a purser in a sea battle, a gamester compared to a merchant-venturer), but these came from rare keywords L1 weighted weakly or dropped, not from reading context. The other 4 were real merchant passages, and L2 got 3 of those right. My inference is that L2 is more robust to label errors because it uses context, while L1's near-perfect score largely rewards rebuilding the rule. This rests on 18 hand-read chunks, so it is suggestive, not proof.

**Historical payoff.** 

I would pick L2. L1 hands our seed list back to us. L2 surfaces vocabulary we did not put in (commodities, exchange, goods, wares, london) and a clear contrast with religious and military discourse, which is what we would want to study historically.
