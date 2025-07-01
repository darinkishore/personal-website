---
author: Darin Kishore
pubDatetime: 2025-07-01T12:00:00Z
title: DSPy is to LLM Engineering What Rust is to Systems Programming
slug: dspy-rust-llm-engineering
featured: false
draft: false
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



<tangent>
^ re above though:
Vibes can get you enough to get users to get evals, but dear lord you should be testing and running your shit in prod. if you don't have live monitoring and testing and you make money off your llm stuff WHAT ARE YOU DOING BRO FIX THAT FUCKING RECORD THINGS IN PROD BRO THE LONGER U WAIT THE MORE MONEY AND DATA YOU LOOOSE
</tangent>

### Making Invalid States Unrepresentable

#### Prompts

**Every prompt is technical debt accruing at the rate of model improvement.**

eg: the switch from non-reasoning to reasoning models. had to rethink almost all prompts, needed to be more declarative. you have to instruct o1 along which dimensions to reason along, because it's weaker—o3 generally just gets things a lot easier. whatever.

basically,
if you work at the prompt layer,

because your instructions, formatting are so tightly tied to the model it's built for—
it becomes harder to change the model and benefit from model capabilities! (unless you have evals—in which case, good for you. keep going.)

But even if you do, it's hard to refactor unless you're principled enough to not put anything other than single tasks in promps. As soon as you do that, you entangle all your concerns, and, well, yeah. shit bro idk.

dspy's optimizers are nice here, but even the system design it forces works wonders.

speaking of,

#### Signatures

simplest sig: `question -> answer`, `questions: list[str] -> answers: list[str]`. pattern extends to all python objects, in -> out. can specify instruction as well.

"signatures make invalid states (ie: not managing context, all the other prompt mgmt bullshit) unrepresentable".

models are now happily good enough where this will be an approach that "just werks" if u break things up enough and follow the decomp low.

so like…
if we buy the decomposition law,

(which… seems to be a fair bit of research confirming LM performance craters the more rules you add in a call, and different prompt formatting strategies have outsized performance—read the canonical darin-dump of the law and you'll note that both complexity/unnaturalness/difficulty in input AND output compromise model perf, which does seem to somewhat empirically align with the research (according to a very short [deep research query](https://chatgpt.com/s/dr_68638dfa3d48819190f5982514d3fed3) hehe)).


 **DSPy's signature abstraction forces us to abide by it**. it forces you to think about LM programs THE WAY YOU SHOULD THINK ABOUT LM PROGRAMS. THESE SHITS ARE UNRELIABLE.DECOMPOSE. DO ONE THING AT A TIME.

the god-prompt does not exist in signature land (assuming you're trying not to. all abstractions are leaky.)

ok.ok.ok. assuming a 95% chance of success for every call (where 5% are some form of failures, and that failures propogate), a 5-step pipeline has uh like fuckin 66% (number made up on the spot).

#### Technical Investment

Okay, say we buy all the DSPy shit. What does this get us ?

The abilitiy to progressively and targetedly inject human intuition into the layers of your pipeline that need it the most. Need some engineering taste for creating good diagrams? Add a `taste` input for each class of diagrams, write some opinions, and ditch it in two model jumps when they r just very tasteful lol

You can progressively REMOVE parts of DSPy programs! (small note: this is bc optimization towards arbitrary metric exists, and you need evals to do this.)

So as model capabilities grow, tasks which you used to have to break up bit by bit will be able to be done better in just a single signature—simplifying your program with time! That's p fire.

(note: need 2 b concrete.)

### Call to Action

(IMPORTANT NOTE: ACKNOWLEDGE THE TRADEOFFS!!!! now that we r done!!!!)

(tbd)

Honestly,

just.. please spend more time working on the abstractions. put more energy into creating optimizers, fucking with signatures in different ways, idk. just

do better on LLM engineering pls

a fun note: this doesn't even begin to get into data ; : )
oh god there is so much here

---

#### Detours

ie: maintaining LLM systems is an incredibly difficult task that… only gets easier with time? But to create reliable LLM systems today, you need to decompose properly.

look at this graph bro:
![[CleanShot 2025-06-29 at 15.38.41@2x.png]]


To make the initial gains on any systems at the frontier of model capabilities, you need some human intuition rn. ok, say we want to make that superhuman. we loosen the constraints strategically. hm. not a good detour.

---
