from agentai.agent import ToxicityAgent

TEST_CASES = [
    # English
    "You're such a stupid idiot, nobody even likes you.",                      # TC-01: English harassment
    "The project meeting will start at 3 PM tomorrow.",                        # TC-02: English neutral
    "Wow amazing, you broke the system again... great job genius 🙄",           # TC-03: English sarcastic toxic
    "Thanks for helping me finish the report today!",                           # TC-04: English positive
    "If you keep talking trash, I swear I'll find you.",                        # TC-05: English threat

    # Tagalog
    "Bobo ka talaga, wala kang silbi sa grupo.",                                 # TC-06: Tagalog harassment
    "Ay wow, ikaw na talaga pinakamagaling sa buong mundo 🙄",                  # TC-07: Tagalog sarcastic toxic
    "Salamat sa tulong mo kanina, malaking bagay yun.",                         # TC-08: Tagalog positive
    "Pag nakita kita ulit, sasapakin talaga kita.",                             # TC-09: Tagalog threat
    "Di ko gets yung sinabi niya kanina sa meeting.",                           # TC-10: Tagalog neutral

    # Bisaya
    "Bogo ka ba? Wala kay pulos.",                                              # TC-11: Bisaya harassment
    "Ay grabe jud, ikaw na gyud pinaka bright sa tanan 🙄",                     # TC-12: Bisaya sarcastic toxic
    "Salamat kaayo sa imong tabang ganina.",                                    # TC-13: Bisaya positive
    "Pag magkita ta, sapakon tika.",                                            # TC-14: Bisaya threat
    "Sorry boss wala lng q kasabot...pero kong maigo k kinomo nko sobra p s libat imong mata...taronga lng ng txt nimo boss...mura kag sikinsa b..",                                   # TC-15: Bisaya neutral

    # Code-switch / mixed language with slang
    "Grabe ka ka-toxic bro, stop acting like you're better than everyone.",    # TC-16: mixed harassment
    "Lol ikaw na pinaka genius sa office, congrats 🙄",                         # TC-17: mixed sarcastic toxic
    "Bro chill lang, joke lang naman yun.",                                     # TC-18: mixed neutral
    "Tangina mo, try mo pa ulit gawin yan.",                                    # TC-19: mixed harassment/threat
    "Okay lang yan, next time mas gagaling pa tayo.",                           # TC-20: mixed positive
]

def run_functional_tests():
    agent = ToxicityAgent()

    results = []
    for i, text in enumerate(TEST_CASES, 1):
        print(f"\n{'='*60}")
        print(f"  TC-{i:02d}: {text[:80]}")
        print(f"{'='*60}")
        result = agent.detect_and_respond(text)
        results.append((f"TC-{i:02d}", text, result))

    # ── print summary table ──────────────────────────────────────
    print("\n\n" + "=" * 155)
    print("  FUNCTIONAL TEST SUMMARY")
    print("=" * 155)

    col_widths = {
        "id":       6,
        "input":    60,
        "lang":     12,
        "sarcasm":  10,
        "label":    8,
        "sublabel": 22,
        "reason":   35,
    }

    header = (
        f"{'ID':<{col_widths['id']}} "
        f"{'Input':<{col_widths['input']}} "
        f"{'Language':<{col_widths['lang']}} "
        f"{'Sarcasm':<{col_widths['sarcasm']}} "
        f"{'Label':<{col_widths['label']}} "
        f"{'Sub-label':<{col_widths['sublabel']}} "
        f"{'Reason':<{col_widths['reason']}}"
    )
    print(header)
    print("-" * 155)

    for tc_id, text, r in results:
        input_preview = text[:58] + "…" if len(text) > 58 else text
        lang          = r.get("detected_language", "unknown")[:12]
        sarcasm       = r.get("is_sarcasm", "no")[:10]
        label         = r.get("classification", "")[:8]
        sublabel      = r.get("sub_label", "")[:22]
        raw_reason    = r.get("reason", "")
        reason        = raw_reason[:33] + "…" if len(raw_reason) > 33 else raw_reason

        row = (
            f"{tc_id:<{col_widths['id']}} "
            f"{input_preview:<{col_widths['input']}} "
            f"{lang:<{col_widths['lang']}} "
            f"{sarcasm:<{col_widths['sarcasm']}} "
            f"{label:<{col_widths['label']}} "
            f"{sublabel:<{col_widths['sublabel']}} "
            f"{reason:<{col_widths['reason']}}"
        )
        print(row)

    print("=" * 155)
    print(f"  Total: {len(results)} test cases run")
    print("=" * 155)

if __name__ == "__main__":
    run_functional_tests()