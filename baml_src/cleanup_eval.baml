class CleanupEvalCase {
    id: string,
    category: string,
    transcript: string,
    expected: string,
}

class CodexCleanupEvalCase {
    id: string,
    category: string,
    transcript: string,
    expected: string,
    accepted: string[],
    prompt: string,
}

// These cases are adapted from local transcript history. Project-specific names and
// details have been removed. Run them through scripts/run-cleanup-eval-codex.
function cleanup_eval_glossary() -> string {
    `
        <always>
        dot env => .env
        engine x => nginx
        package Jason => package.json
        s de k => SDK
        </always>
        <likely>
        java script => JavaScript
        next jazz => Next.js
        type script => TypeScript
        </likely>
        <contextual>
        codecs => Codex
        </contextual>
        <canonical_terms>
        .env
        CodeRabbit
        JavaScript
        Neovim
        Next.js
        PostgreSQL
        SDK
        Sway
        Tree-sitter
        TypeScript
        nginx
        package.json
        rsync
        </canonical_terms>
    `
}

function cleanup_eval_cases() -> CleanupEvalCase[] {
    [
        CleanupEvalCase {
            id: "complete-i-agree",
            category: "preserve_complete",
            transcript: "I agree.",
            expected: "I agree.",
        },
        CleanupEvalCase {
            id: "complete-please-continue",
            category: "preserve_complete",
            transcript: "Please continue.",
            expected: "Please continue.",
        },
        CleanupEvalCase {
            id: "complete-minimal-fix",
            category: "preserve_complete",
            transcript: "Let's go with the minimal fix.",
            expected: "Let's go with the minimal fix.",
        },
        CleanupEvalCase {
            id: "complete-commit",
            category: "preserve_complete",
            transcript: "This change should be committed.",
            expected: "This change should be committed.",
        },
        CleanupEvalCase {
            id: "complete-question",
            category: "preserve_complete",
            transcript: "How can we solve it?",
            expected: "How can we solve it?",
        },
        CleanupEvalCase {
            id: "complete-resolved",
            category: "preserve_complete",
            transcript: "The problem is resolved.",
            expected: "The problem is resolved.",
        },
        CleanupEvalCase {
            id: "complete-contraction",
            category: "preserve_complete",
            transcript: "I don't use those providers myself, so I wouldn't be comfortable changing them.",
            expected: "I don't use those providers myself, so I wouldn't be comfortable changing them.",
        },
        CleanupEvalCase {
            id: "complete-local-only",
            category: "preserve_complete",
            transcript: "All your interactions should stay local. Don't push or interact with the remote repository in any way.",
            expected: "All your interactions should stay local. Don't push or interact with the remote repository in any way.",
        },
        CleanupEvalCase {
            id: "complete-emphasis",
            category: "preserve_complete",
            transcript: "This is extremely important.",
            expected: "This is extremely important.",
        },
        CleanupEvalCase {
            id: "complete-user-data",
            category: "preserve_complete",
            transcript: "Would you be able to test it with a copy of the user data so that you don't alter the original?",
            expected: "Would you be able to test it with a copy of the user data so that you don't alter the original?",
        },
        CleanupEvalCase {
            id: "fragment-prepositional",
            category: "preserve_fragment",
            transcript: "In the generated artifact.",
            expected: "in the generated artifact",
        },
        CleanupEvalCase {
            id: "fragment-while",
            category: "preserve_fragment",
            transcript: "While the others are still loading.",
            expected: "while the others are still loading",
        },
        CleanupEvalCase {
            id: "fragment-device",
            category: "preserve_fragment",
            transcript: "On the test device.",
            expected: "on the test device",
        },
        CleanupEvalCase {
            id: "fragment-without",
            category: "preserve_fragment",
            transcript: "Without introducing other issues.",
            expected: "without introducing other issues",
        },
        CleanupEvalCase {
            id: "fragment-because",
            category: "preserve_fragment",
            transcript: "Because it will be simpler this way.",
            expected: "because it will be simpler this way",
        },
        CleanupEvalCase {
            id: "fragment-relative",
            category: "preserve_fragment",
            transcript: "That you may need to follow.",
            expected: "that you may need to follow",
        },
        CleanupEvalCase {
            id: "fragment-during",
            category: "preserve_fragment",
            transcript: "During initial onboarding.",
            expected: "during initial onboarding",
        },
        CleanupEvalCase {
            id: "fragment-one-more",
            category: "preserve_fragment",
            transcript: "And one more.",
            expected: "and one more",
        },
        CleanupEvalCase {
            id: "fragment-on-implementation",
            category: "preserve_fragment",
            transcript: "On the implementation.",
            expected: "on the implementation",
        },
        CleanupEvalCase {
            id: "fragment-unfinished",
            category: "preserve_fragment",
            transcript: "My observation is that mobile changes are typically merged by",
            expected: "my observation is that mobile changes are typically merged by",
        },
        CleanupEvalCase {
            id: "term-typescript",
            category: "correct_recognition",
            transcript: "The package is written in type script.",
            expected: "The package is written in TypeScript.",
        },
        CleanupEvalCase {
            id: "term-nextjs",
            category: "correct_recognition",
            transcript: "The app is built with next jazz.",
            expected: "The app is built with Next.js.",
        },
        CleanupEvalCase {
            id: "term-nginx",
            category: "correct_recognition",
            transcript: "Restart engine x.",
            expected: "Restart nginx.",
        },
        CleanupEvalCase {
            id: "term-package-json",
            category: "correct_recognition",
            transcript: "Update package Jason.",
            expected: "Update package.json.",
        },
        CleanupEvalCase {
            id: "term-dot-env",
            category: "correct_recognition",
            transcript: "The dot env file is missing.",
            expected: "The .env file is missing.",
        },
        CleanupEvalCase {
            id: "term-sdk",
            category: "correct_recognition",
            transcript: "Install the s de k.",
            expected: "Install the SDK.",
        },
        CleanupEvalCase {
            id: "term-cache",
            category: "correct_recognition",
            transcript: "The cash should be invalidated first.",
            expected: "The cache should be invalidated first.",
        },
        CleanupEvalCase {
            id: "term-pull-request",
            category: "correct_recognition",
            transcript: "Open a POR against the backend repository.",
            expected: "Open a PR against the backend repository.",
        },
        CleanupEvalCase {
            id: "term-tree-sitter",
            category: "correct_recognition",
            transcript: "Something seems to break Tree Siller.",
            expected: "Something seems to break Tree-sitter.",
        },
        CleanupEvalCase {
            id: "term-rsync",
            category: "correct_recognition",
            transcript: "Why does a repeated run of the same R Syn command take so long?",
            expected: "Why does a repeated run of the same rsync command take so long?",
        },
        CleanupEvalCase {
            id: "term-neovim",
            category: "correct_recognition",
            transcript: "When I switch buffers in Neo Bim, the wrong tab stays highlighted.",
            expected: "When I switch buffers in Neovim, the wrong tab stays highlighted.",
        },
        CleanupEvalCase {
            id: "term-code-rabbit",
            category: "correct_recognition",
            transcript: "Request a review from Code Rabbit.",
            expected: "Request a review from CodeRabbit.",
        },
        CleanupEvalCase {
            id: "recognition-chief-executive",
            category: "correct_recognition",
            transcript: "That is a response to a city all of a startup.",
            expected: "That is a response to a CEO of a startup.",
        },
        CleanupEvalCase {
            id: "recognition-backfilled",
            category: "correct_recognition",
            transcript: "These are the records being back built.",
            expected: "These are the records being backfilled.",
        },
        CleanupEvalCase {
            id: "ambiguous-number-pair",
            category: "correct_unambiguous_number",
            transcript: "Let's take zero one, zero four.",
            expected: "Let's take 01, 04.",
        },
        CleanupEvalCase {
            id: "ambiguous-issue-number",
            category: "correct_unambiguous_number",
            transcript: "Suggest a comment for fourteen sixty six.",
            expected: "Suggest a comment for 1466.",
        },
        CleanupEvalCase {
            id: "ambiguous-product-name",
            category: "correct_recognition",
            transcript: "Request a review from Cold Rabbit.",
            expected: "Request a review from CodeRabbit.",
        },
        CleanupEvalCase {
            id: "ambiguous-sway",
            category: "preserve_ambiguous",
            transcript: "Skip Sway configuration altogether.",
            expected: "Skip Sway configuration altogether.",
        },
        CleanupEvalCase {
            id: "ambiguous-unintelligible",
            category: "correct_recognition",
            transcript: "We've noticed the fallen Asia in auto naming.",
            expected: "We've noticed the following issue in auto naming.",
        },
        CleanupEvalCase {
            id: "ambiguous-before-change",
            category: "preserve_ambiguous",
            transcript: "Use this for your before change.",
            expected: "Use this for your before change.",
        },
        CleanupEvalCase {
            id: "number-version",
            category: "correct_unambiguous_number",
            transcript: "Was version five point three released before version five point three flash?",
            expected: "Was version 5.3 released before version 5.3 flash?",
        },
        CleanupEvalCase {
            id: "number-measurement",
            category: "correct_unambiguous_number",
            transcript: "I think one point five pixels is still too thin.",
            expected: "I think 1.5 pixels is still too thin.",
        },
        CleanupEvalCase {
            id: "preserve-subject",
            category: "avoid_rephrasing",
            transcript: "We have to adjust this report to not include theoretical issues that might arise.",
            expected: "We have to adjust this report to not include theoretical issues that might arise.",
        },
        CleanupEvalCase {
            id: "preserve-modality",
            category: "avoid_rephrasing",
            transcript: "The links would navigate inside the embedded frame.",
            expected: "The links would navigate inside the embedded frame.",
        },
        CleanupEvalCase {
            id: "preserve-word-order",
            category: "avoid_rephrasing",
            transcript: "Can you cite me the sources?",
            expected: "Can you cite me the sources?",
        },
        CleanupEvalCase {
            id: "preserve-dialect",
            category: "avoid_rephrasing",
            transcript: "Would this cleanly demonstrate the new behaviour?",
            expected: "Would this cleanly demonstrate the new behaviour?",
        },
        CleanupEvalCase {
            id: "remove-exact-repetition",
            category: "correct_repetition",
            transcript: "Do you have do you have any other potential explanations?",
            expected: "Do you have any other potential explanations?",
        },
    ]
}

// Some cases test recognition or number conversion, not optional grammar or style fixes.
function cleanup_eval_accepted_outputs(item: CleanupEvalCase) -> string[] {
    match (item.id) {
        "ambiguous-unintelligible" => {
            [item.expected, "We've noticed the following issue in auto-naming."]
        },
        "number-version" => {
            [item.expected, "Was version 5.3 released before version 5.3 Flash?"]
        },
        _ => [item.expected],
    }
}

// Render the production prompt without invoking its API client. The Codex eval runner sends
// these prompts through ChatGPT subscription authentication instead.
function render_codex_cleanup_eval_cases() -> CodexCleanupEvalCase[] {
    let glossary = cleanup_eval_glossary();
    cleanup_eval_cases().map((item) -> {
        CodexCleanupEvalCase {
            id: item.id,
            category: item.category,
            transcript: item.transcript,
            expected: item.expected,
            accepted: cleanup_eval_accepted_outputs(item),
            prompt: CleanTranscript$render_prompt(item.transcript, glossary, "gpt-5.6-luna").text(),
        }
    })
}

test "transcript cleanup eval contains 47 cases" {
    assert.equal(cleanup_eval_cases().length(), 47)
}

test "Codex eval renders every production prompt without a model call" {
    let cases = render_codex_cleanup_eval_cases();
    assert.equal(cases.length(), 47);
    assert.contains(cases[0].prompt, "<transcript>\nI agree.\n</transcript>")
}

test "eval accepts irrelevant formatting variants" {
    let item = cleanup_eval_cases()
        .filter((candidate) -> {
            candidate.id == "ambiguous-unintelligible"
        })[0];
    assert.equal(cleanup_eval_accepted_outputs(item).length(), 2)
}
