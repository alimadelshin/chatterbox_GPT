# chatterbox_GPT

1. Download data (https://files.pushshift.io/reddit/comments/) and unpack.
2. Drop everything into the get_data folder.
3. Run mining.py, pre-registering in the timeframe the name of one previously extracted json file (example: timeframe = 2020-01). Using this program, we will get a database file from a json file, in which each parent-comment refers to its child-comment with the best score, and vice versa.
4. Let's go drink tea.
5. Run db_to_text.py, filling in everything in advance.
6. Repeat steps 1-5 until you feel like you have enough data. (Yes, you can process everything in one pass, but this is too expensive, because you need terabytes of free space)
7. Run sep_to_pairs.py, which will collect all the data in three files, dividing everything into pairs parent -> (child, score). We won't need the score during training, but it may be useful in the future.
8. Transfer the received 3 txt files to data_processing / data.
9. Run grammizer.py to find the most popular n-gramms. If RAM problems occur, decrease limit_value and buffer.
10. Run norm.py, which will reduce the amount of our data, while trying not to lose in variety. The higher the limit, the less data we get at the output.
11. Run shuffle.py to shuffle our examples.
12. During training, we need some RAM, so it would be nice to divide our examples into 5-10 parts, naming them (q1, q2, q3 ... r1, r2, r3 ...) and edit line 85 in the learning file. py to (0, number of resulting txt files / 2).
13. Start learning.py, having previously adjusted the parameters of the neural network in the program.
14. Go to drink tea.
15. Go to the store.
16. Have a family and children.
17. Retire.
18. Run eval.py and evaluate the results.
