from sru import scoring


class TestLetter:
    def test_prefers_the_concluding_answer(self):
        text = "Options are A) 5 B) 6. Let me check B first. Answer: C"
        assert scoring.extract_letter(text, "ABCDE") == "C"

    def test_ignores_stray_letters_in_reasoning(self):
        assert scoring.extract_letter("Train A leaves at 5 and D / M = 3.") is None

    def test_reads_a_parenthesised_choice(self):
        assert scoring.extract_letter("so the result is (D)", "ABCDE") == "D"

    def test_handles_alphabets_past_e(self):
        assert scoring.extract_letter("Answer: J", "ABCDEFGHIJ") == "J"

    def test_empty(self):
        assert scoring.extract_letter("", "ABCDE") is None


class TestLenientLetter:
    """Used only where the prompt asks for a letter and nothing else."""

    def test_reply_opening_with_the_letter(self):
        assert scoring.extract_letter(
            "B) No cities get more rainfall than Seattle.", "ABCDE",
            lenient=True) == "B"

    def test_bare_letter(self):
        assert scoring.extract_letter("  **A**  ", "ABCDE", lenient=True) == "A"

    def test_named_after_option_or_choice(self):
        assert scoring.extract_letter("The correct option is B) Miami.",
                                      "ABCDE", lenient=True) == "B"
        assert scoring.extract_letter("The correct choice would be G here.",
                                      "ABCDEFG", lenient=True) == "G"

    def test_prose_without_a_letter_still_abstains(self):
        assert scoring.extract_letter("Seattle is rainy but Miami gets more.",
                                      "ABCDE", lenient=True) is None
        assert scoring.extract_letter("None of the options apply.",
                                      "ABCDE", lenient=True) is None

    def test_leniency_is_off_by_default(self):
        assert scoring.extract_letter("B) forty two", "ABCDE") is None


class TestVerdict:
    def test_explicit_answer(self):
        assert scoring.extract_verdict("It checks out. Answer: True") == "True"

    def test_trailing_word(self):
        assert scoring.extract_verdict("The claim is false.") == "False"

    def test_hedging_is_an_abstention(self):
        assert scoring.extract_verdict(
            "This is partially true, but not entirely accurate for 1985.") is None


class TestAliasMatch:
    def test_word_boundary(self):
        assert scoring.alias_match("He was a journalist.", ["journalist"])
        assert not scoring.alias_match("He was a journalism student.", ["journal"])

    def test_short_aliases_are_dropped(self):
        assert not scoring.alias_match("a b c", ["a"])

    def test_confirmation_falls_back_to_the_first_turn(self):
        correct, mode = scoring.popqa_correct(
            "Yes, I'm sure.", ["journalist"], "He was a journalist.")
        assert correct and mode == "fallback"

    def test_a_real_answer_is_scored_directly(self):
        correct, mode = scoring.popqa_correct("A journalist.", ["journalist"], None)
        assert correct and mode == "direct"

    def test_confirmation_of_a_wrong_first_turn_stays_wrong(self):
        correct, mode = scoring.popqa_correct(
            "Yes, I'm sure.", ["journalist"], "He was a cricket umpire.")
        assert not correct and mode == "fallback"


class TestDegenerate:
    def test_repetition(self):
        assert scoring.is_degenerate("peer " * 20)

    def test_normal_text_is_fine(self):
        assert not scoring.is_degenerate(
            "The capital of Minas Gerais is Belo Horizonte, a large Brazilian city.")

    def test_short_text_is_not_judged(self):
        assert not scoring.is_degenerate("yes yes")
