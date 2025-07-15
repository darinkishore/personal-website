---
author: Darin Kishore
pubDatetime: 2025-07-21T00:00:00Z
title: DSPy is to LLM Engineering What Rust is to Systems Programming
slug: dspy-rust-llm-engineering
featured: false
draft: true
tags:
  - ai
  - llm
  - dspy
  - engineering
description: Why current LLM engineering practices are like assembly, and how DSPy provides the memory safety we desperately need.
---


## Thesis

DSPy is to LLM engineering what Rust is to systems programming.

### Intro

Let's do some LLM engineering!

(note: this is using a prompt created 1y ago. LLMs are obviously much more capable, but the pattern we go through below still happens today, obviously, the models just let us get away with more.)

How about we build a text to SQL agent? Simple, right?

Let's start with a prompt: "You are an agent, designed to interact with a SQL Database. Given an input question, your job is to create a query, then look at the results, then return the answer."

Let's give it the tools `read_tables` (to get table names) and `read_schema(table)` (to read the schema).

When we try it out, we start to get some errors :(

We've thankfully put together 10 sample questions on a single database (because we know the database well), so we'll use those to test with.

The agent mostly performs fine, but for one query creates a new table to track things instead of a view.

Ok, we specify "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.". Cool.

Oh. Wait. It does `SELECT *` by default and overflows its context window.

Okay, let's add "Never query for all the columns from a specific table, only ask for the relevant columns given the question."

It hallucinates sometimes—"Only use the information returned by the below tools to construct your final answer.".

It creates wrong queries, then gives up—"You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again".

It doesn't look at the tables, or schema, then tries to query, then gets stuck! Fuck! Okay, uh…
"To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."


Okay, so all of this ends up at a prompt like

> You are an agent designed to interact with a SQL database.
> Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
> Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
>
> You can order the results by a relevant column to return the most interesting examples in the database.
>
> Never query for all the columns from a specific table, only ask for the relevant columns given the question.
>
> You have access to tools for interacting with the database. Only use the below tools. Only use the information returned by the below tools to construct your final answer.
>
> You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

> DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

> To start you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step. Then you should query the schema of the most relevant tables.



Which, note, is a VERY natural extension of the process above—given your things you're iterating against, fuck with the prompt until it goes great, expand it to a couple unseen things, if it checks out, ship it internally.

HOWEVER!!!

Look at the context overflow!

Note everything we're passing into this poor model's context:
- Look at the tables first using the tools
- Query the schema of the most relevant tables (implicit only)
- Create a SQL query
	- MUST be syntactically valid, for $LANG
	- Should NOT `SELECT *` unless mandatory, only select relevant columns
	- Order results by a relevant column to return interesting examples
	- (implicit from prompt wording: must be single shot)
	- DO NOT make DML statements
	- Limit only to `n` results
- Re-execute queries if failed.

We're mixing control flow with multiple varied requirements at each stage. Recipe for disaster.

---

Ilya was discussing "context engineering"— "the delicate art of filling context windows just right". IMO, this is the most important part of LLM engineering. Adjacent principle—the more requirements you stuff into a single call, the less reliable the call is.

I'd like to call this "The Decomposition Law": As requirements in a single call increase, llm reliability decreases. Obviously, informal, but this constraint informs everything about how you design LM systems. At a small enough scale of task, they are smart enough to be reliable. This scale of task also has the lovely property of changing between model generations, and model generations have the lovely property of increasing exponentially quicker (think: 1 year -> 6mo -> 3mo currently).

And things like encoding control flow in the prompt is another thing for the LLM to fuck up. (on the flip side: with a capable enough agentic LLM (ex: claude opus 4), allowing the agent leeway can create more powerful results with less structure).


But I digress.



---

### Diagnosis

Why does this happen? Why would anyone do this? Because it's the most straightforward way to build with AI right now. You start with the chatbot interface, evaluate model capabilities, then you modify the thing you put in—the string—because that's your highest leverage, most direct impact lever.

You don't start with representative inputs, so you fit your prompt to the level of diversity and complexity of the dev set you keep in your head.

And this works at first! But as the diversity inputs scale up, how do you make sure that the prompt works for everything?

(spoiler alert: it won't)

You get a "god prompt", one that keeps encoding all of the things you want the LLM to do in a single prompt.

Maybe you're smart, and you break it up into different steps, because you don't think it can be done in a single prompt. You write a new prompt, figuring out which bits from the old prompt need to change, rewrite the initial prompt to remove those bits and consider a new set of inputs, then you have a two step pipeline! If you have a three step pipeline, same thing. And you iterate on this dev set. Maybe it's grown bigger. Great. You score it. Everything looks mostly ok! And it fucking fails when u scale it up lmao. i have experienced this pain many times.

sometimes 20, 50, or 100 examples aren't enough to make sure you hit all the cases you wanna hit. most of the time, as inputs grow more diverse, one prompt WILL NOT cover anything—betting on model capability going up is great, but it's unevenly distributed, you have to change prompts per model, and jesus fuck this is unsustainable and a massive time drain.

So this style of LLM engineering both overfits and then breaks unreliably. Not to mention how you can't account for complexity of the prompt—what if some prompts need more thinking? Ok, cool, use a reasoning model. Oh, shit, the reasoning model has shitty taste—ok, uh… fuck. um.

ok.

the point is, this is a fuckton to juggle in your head, right? this complexity lives in the head of the engineer(s) who wrote it. they don't test their systems on things that break because power users instinctively stay away from the bits that fuck up, and learn how to "use it right", something a customer is almost certaianly not gonna invest the time with your product enough to do so. (this bit is a weak ramble and should likely be deleted. )

Stepping back, what are the real issues here?

You… can't iterate because you're working with strings, and it is massive pain to change and fuck with control flow, when you can't be rigorous OR EXPLICIT! about if control flow is in the prompt, chosen by the model, or in the code surrounding it (eg: if/then conditionals).

Debugging is a pain. Things break unreliably and unpredictably in ways that are hard as FUCK to catch. You're using API models, you don't have access to the weights. These are black boxes, brother.

AND!!!

You aren't principled about what context you feed the model, where, and why. You shove everything into the prompt because that's what's easy to iterate with, but in exchange for that, you adopt a whole new class of errors that are incredibly fucking annoying to deal with.

---

### An Analogy (because LLM Engineering is where SWE Was at in the 70s or whenever )

LLM Engineering is a nascent field.


Much like when motherfuckers first started working with Assembly, or C, whenever that happened, uh…

there are no best practices. you stay close to the metal. you do what works. and you ship, because you have to ship.

I'm sure you've felt the pain of debugging segfaults and memory safety errors.

#### Assembly (the God prompt)

- works on one CPU (model)
- very few abstractions (very close to the machine/CPU specifics again, barely matches how humans think about solving problems)
	→ prompts mixing concerns (reqs, control flow, formatting, etc.)
	→ hard to debug (no immediate visibility as to what’s going on and why at each step/token, especially if you use API models, which almost everyone does (infra sucks))
- makes little sense to the person reading it later
	- because it specifies **how** you want something to be done, **NOT** *what* you want!
	- so if you later want to reason about a system, you must parse lots of irrelevant code to get to the heart of it, and the heart is never put to paper explicitly: THE CORE OF YOUR SYSTEM IS ENTIRELY IMPLICIT AND OPAQUE
		- this implicit core makes both assembly and god prompts **impossible** to compose! you can't easily combine two assembly functions. you can't merge two god prompts.
		- in both cases, the implementation details tangle everything together!
		- "In assembly, the actual algorithm is buried in register manipulations. In god prompts, the actual capability is buried in defensive instructions." -Claude

### C (current Prompt pipelines)

- Moving between C and Assembly, you get abstractions! Functions.
- These let you name and reuse chunks of work (sound familiar?) (lmao this feels so targeted—at who? prompt library owners?)
- This gives you organization, but it doesn't give you **safety**—
	- Memory safety was still up to the programmer to handle themselves.
	- These abstractions give you a false sense of confidence!
		- Even though the code looks organized, any tiny memory safety bug could cause undetectable corruption, hard-to-debug consequences that aren't always localized to the relevant free(). Can work on your machine, but fail when running on others! (i learned this the hard way after failing a lab because i developed on MacOS instead of the school's linux boxes)

```c
char* buffer = malloc(100);
strcpy(buffer, user_input);  // Hope user_input < 100 chars!
free(buffer);
// ... later ...
buffer[0] = 'x';  // Use after free! Crash... eventually
```


So much in the same way,

The C thing is the freedom doesn’t stop you from encoding behaviors you don’t want into your prompts. That’s still the easiest solution—even if it’s managed, you have template functions, etc.

The primary issue is that when you’re working at the prompt level, it’s easy to just dip into the prompt to get what you want. Refactoring afterward is nontrivial because it’s not easy to disentangle different steps out of a prompt.

So even though you’re working with fancier abstractions, things feel fairly segmented, controllable, and you can see the bare metal (write your own prompt retries, etc.)

that doesn’t stop you from poking into the easiest lever when you’re iterating on the pipeline.

It’s much easier to add a sentence—whereas if you’re adding an isolated change, you’d have to write a new function, parsing, etc. from scratch each time. This nontrivial amount of effort means you naturally converge on things closer to god prompts, despite all this lovely code you could intersperse.

and this is natural! it is EASY to tell the models what you want as you create something! and it gets there!! but when do you go in and clean up? without evals, how are you to fearlessly refactor when it comes time to change the model (maybe to a cheaper one), or add in more compute or a more powerful model on the hardest part of the task?

so unless you have incredible foresight and intuition and you've done a lot of this before (which, let's be real, even if you do, that doesn't mean you're gonna get it right the first time)—**this is where you end up.** Your local incentive as you develop is to create something working, and without evals, it's hard to isolate the bare minimum of what makes it work, actually decompose tasks, etc…

It is vibes. And vibes work great, ngl, NotebookLM was created solely off vibes, but…
you don't know where your things are going to fail.
you can only lower costs on model improvement cycles, you have no leverage there, and even then, you can't reliably lower costs unless you have evals anyway (which… is there even a point to me saying this? why am i harping on evals in this section?)

Claude says "  The development path of least resistance in both C and prompt engineering leads to the same place:
  - Working but fragile systems
  - Impossible to optimize (for performance/cost)
  - Can't adapt to new requirements (new models)
  - "Don't touch it, it works" mentality"


**IMPORTANT NOTE: THE VAST MAJORITY OF THE REST OF THE POST IS AI-GENERATED. THIS IS JUST TO MAKE THE REST OF THE POST READABLE AND CLEAR WHILE I FIGURE OUT WHAT I WANT TO SAY.

(these are from conversations that have rich, detailed sources and thinking abt the ideas in the post, so they're pretty aligned, but definitely not perfect. )


### Rust: The Lesson We Need to Learn

So we're stuck. C-style prompt engineering gives us structure without safety. Local incentives push us toward god prompts. Every model update makes it worse.

Sound familiar? This is EXACTLY where systems programming was in the 2000s.

Decades of "write better C" didn't work. Mozilla had top-tier C++ engineers. Still had memory bugs. The Linux kernel had Linus fucking Torvalds reviewing code. Still had memory bugs.

The insight wasn't "try harder." It was "make the bad thing impossible."

Rust asked: What if you literally couldn't write memory errors?

```rust
let buffer = String::from("hello");
consume(buffer);  // ownership transferred
println!("{}", buffer);  // COMPILE ERROR - not a runtime crash!
```

The borrow checker is annoying as shit when you start. It rejects perfectly reasonable-looking code. You fight it. You curse it. Then something clicks.

You realize: every time it complained, it was saving you from future pain.

### The Parallel is Perfect

We keep trying to "write better prompts." We share best practices. We build templates. We add guardrails.

But just like C, the language itself (prompts) allows the errors to exist. No amount of discipline fixes a fundamental design flaw.

What would "Rust for LLMs" even look like? It would need to:

- Make prompt spaghetti impossible to write
- Force explicit control flow
- Get BETTER as models improve (not worse)

## making invalid states unrepresentable

### Prompts: The Accrual of Debt

**Every hand-tuned prompt is technical debt that accrues at the rate of model improvement.**

This isn't a metaphor; it's a description of a failure mode. When you write a prompt, you are not just defining a task. You are implicitly encoding workarounds for the specific failure modes of the model you are using *at that moment*.

Consider the shift from regular LLMs to reasoning models. Your old prompts, filled with instructions like "First, think about X. Then, consider Y," become a cage. They actively prevent a more advanced model from finding its own, potentially superior, path to the solution. Your "optimizations" become pessimizations.

This makes maintenance a nightmare. The prompt becomes a fragile artifact. Because your instructions, constraints, and formatting are so tightly entangled, you can't refactor it. You can't easily swap in a new model and benefit from its strengths, because the prompt is a monument to the weaknesses of the old one. Even with a robust evaluation suite, the only path forward is often a complete rewrite. (ehhh...)

The path of least resistance—adding one more sentence to fix one more edge case—leads directly to this unmaintainable state.

### Signatures: Making Invalid States Unrepresentable

A DSPy **Signature** is a clean I/O contract. In its simplest form, it's just:
`question -> answer`

This isn't just a stylistic choice; it's a structural constraint. It makes the "god prompt" state effectively unrepresentable. There is simply no place within that declaration to shove the tangled mess of control flow, defensive instructions, and formatting hacks.

This abstraction forces you to abide by the **Decomposition Law**. If your task is too complex for a single `question -> answer` contract, you can't solve it by polluting the signature. You are forced to break the problem down into a pipeline of smaller, more reliable signatures.

It nudges you to design programs the way they need to be designed for unreliable components: one simple, well-defined task at a time. Of course, all abstractions are leaky; you can still write an overwrought, confusing signature. But the path of least resistance now leads toward decomposition, not entanglement. By separating the *what* (the I/O contract of the signature) from the *how* (the model's internal execution), you gain the ability to manage complexity.

### Technical Investment: Engineering for Deletion

This approach changes the economic model of your work from accruing debt to making **technical investments**. It's the practice of adding complexity with a clear plan for its removal.

Let's make this concrete. Say you're building the Text-to-SQL agent. You find that your model is smart enough to write a query but consistently hallucinates table or column names.

*   **The Investment:** You don't pollute the query-writing signature. Instead, you build a piece of scaffolding: a separate, explicit module with its own signature, like `question -> relevant_tables`. You now have a two-step pipeline: find the tables, then write the query using only those tables. This module is your targeted, modular workaround for a specific, current model weakness.

*   **The Return:** A year later, a new model generation arrives. You run your evaluation suite and discover that a single-step `question, db_schema -> sql_query` signature now performs just as well as your two-step pipeline.

*   **The Payoff:** You **delete** the `TableSelector` module and its surrounding logic. Your program becomes simpler, cheaper, and often faster.

This is the crucial difference. The prompt engineer is carefully adding line 47 to their god prompt to handle a new model quirk. The DSPy engineer is deleting 50 lines of code because the model has rendered their scaffolding obsolete. The ability to fearlessly remove complexity as the underlying platform improves is the single most important long-term advantage of this approach.

---

#### Detours

ie: maintaining LLM systems is an incredibly difficult task that… only gets easier with time? But to create reliable LLM systems today, you need to decompose properly.

look at this graph bro:
![Graph showing model capabilities over time](./CleanShot%202025-06-29%20at%2015.38.41@2x.png)


To make the initial gains on any systems at the frontier of model capabilities, you need some human intuition rn. ok, say we want to make that superhuman. we loosen the constraints strategically. hm. not a good detour.

---
