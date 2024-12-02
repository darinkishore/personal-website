---
author: Darin Kishore
pubDatetime: 2024-12-02T13:26:00Z
title: Building Better AI Tools with MCP
slug: mcp
featured: true
draft: false
tags:
  - ai
  - mcp
description: Lessons learned from building AI tools and how the Model Context Protocol (MCP) is cool.
---


I've been building a lot this semester. Terrible at shipping, trying to change that. This post is my first attempt - sharing what I've learned about building better AI tools, sparked by Anthropic's new Model Context Protocol.

This all started because I took liberal arts classes as a "break" from technical subjects. Five days a week of attendance-required classes, endless papers to write. Busywork pulling me away from what I actually enjoy: building and research.


So I started automating everything I had to do more than twice. Each friction point became a new AI tool. Research assistant here, documentation RAG pipeline there. They worked, but they were isolated solutions.

It's been really nice because it let me do what I love while not failing out of my classes.

A week ago, Anthropic dropped the Model Context Protocol (MCP), and suddenly there's a standard way to build and connect all these tools. After two years of building AI tools, I've learned that context management and tool design matter more than anything else—and MCP finally gives us a standardized way to get both right.


# MCP: A New Standard for AI Integration

The Model Context Protocol (MCP) fundamentally changes how we can build AI tools. At its core, it provides:

1. A standardized way to expose **resources** - whether that's files, database records, or live system data. Your LLM can maintain awareness of available information without being overwhelmed by it. Usually larger documents.

2. **Tools** that let LLMs take actions through well-defined interfaces. Each tool has clear inputs, outputs, and error handling, making it easier to build reliable AI interactions.


What makes this exciting isn't just the individual capabilities - it's that it's an open, standard protocol. This means:
- Tools built by different developers can work together seamlessly
- Best practices can emerge and be shared across the community
- Integration patterns become consistent and reliable
- Many LLM applications will be able to connect to any MCP server to provide arbitrary functionality.

Instead of everyone building their own custom integration patterns, we get a common language for AI interactions. MCP greatly reduces wasted effort. My scattered collection of tools can now work together in a coherent system, and more importantly, they could work with tools built by others.


## Building tools for LLMs

Now that we have a standard way to connect tools, we can focus on making them actually good. Here's what I've learned about designing tools that LLMs love to use.

### Design
Over two years of building LLM tools, I've found one principle holds consistently: quality is directly proportional to how small the scope and how easily-parsed the inputs are. The tool needs to match how LLMs actually process information and tasks.

Think about what an LLM needs to effectively help with research. If you just ask it to "find papers about wellness," you're missing crucial context. What's the purpose of the research? What kind of analysis are you doing? What theoretical framework are you working within? Without this context, you'll get generic results that might be tangentially relevant but don't actually advance your work.

This led me to design around a simple pattern: pair every query with its purpose. Not just what you're looking for, but why you're looking for it. Here's what this looks like in practice:

```python
# note: this is only useful because it uses Exa,
# an embeddings-based web search platform
QueryPurpose(
    purpose='To build a theoretical framework for analyzing wellness industry trends.',
    question='Academic analysis of commodification in the wellness industry',
    queries=[
        ExaQuery(
            text='Here is an academic paper analyzing cultural appropriation in modern wellness industries:',
            category='research paper',
        ),
        ExaQuery(
            text='Here is a scholarly analysis of how luxury brands commodify spiritual practices:',
            category='research paper',
        )
    ]
)
```


When you're brainstorming with Claude about a paper outline, you've discussed your main arguments, potential counterpoints, and where you need more support. When you need to find sources, all of that conversation becomes valuable context.

The `(purpose, query)` pattern lets you capture this context precisely. Maybe you're looking for sources to:
- Back up your argument about wellness commodification
- Find counterarguments you haven't considered
- Explore how other scholars have approached similar analyses

Each search knows not just what you're looking for, but why you're looking for it in the context of your broader conversation and goals. This context awareness then informs how the tool approaches the search - which search strategies to use, how to evaluate results, what kind of sources to prioritize.

It's about maintaining coherent context through different levels of interaction - from high-level project planning all the way down to individual searches. Each layer understands its role in your broader work.

This context awareness becomes even more powerful when combined with Chain of Thought prompting. The model doesn't just passively receive context - it can actively reason about how to best serve your needs at each level.

This explicit reasoning helps the model make better decisions about how to use the tools. When Claude understands both what you need and why you need it, it can choose more appropriate search strategies and prioritize different kinds of results. Because the purpose and the exact query is specified beforehand as well, after the results are returned, it alows the model to evaluate findings more precisely.

Small details in tool design make a big difference in how well LLMs can use them. For example:
- Using XML-style outputs gives clear structure
- Short semantic IDs (like "big-gopher" or "red-fish") are easier for models to track than UUIDs or long identifiers
- Consistent patterns help models build reliable mental models of tool behavior

But good design is just the start. The real magic happens when you let your tools improve through use.


### Building Tools That Learn

Here's something cool: you can build AI tools that get better through use. Not through complex retraining pipelines or massive datasets, but through simple, structured feedback from the models using them.

This is where DSPy comes in. DSPy provides a way to build modular AI systems that can optimize themselves. Instead of tinkering with prompts or managing complex training pipelines, DSPy lets you write clear Python code that describes what you want your LLM to do, then helps you make it better. By turning your interactions with the tool into training data, you make continuous improvement practically automatic.

For instance, my research pipeline is built from two DSPy modules:

```python
# simplified version of the PurposeDrivenQuery module
# `queries` desc has examples + guidelines on creating good exa queries
class PurposeDrivenQuery(dspy.Signature):
    """Generate optimized search queries based on purpose and question."""

    purpose: str = dspy.InputField(
        desc='why do you want to know this thing? (ie: relevant context from your task)'
    )
    question: str = dspy.InputField(
        desc='what do you want to know, more specifically?'
    )
    queries: list[ExaQuery] = dspy.OutputField(
        desc='optimized queries following best practices'
    )

class ExtractContent(dspy.Signature):
    """
    Extract information from search results to get the best results.
    Only keep search results that are relevant to the original query.
    """

    original_query: QueryRequest = dspy.InputField( # just a purpose, query
        desc='The query that generated the search results, providing context for determining relevance.'
    )
    content: SearchResultItem = dspy.InputField(desc='the raw search result')

    cleaned_response: Union[SummarizedContent, None] = dspy.OutputField(
        desc='the cleaned search result'
    )
# both response pydantic types are specific about whats expected

query_generator = dspy.ChainOfThought(PurposeDrivenQuery)
content_cleaner = dspy.ChainOfThought(ExtractContent)

result = query_generator(purpose="Finding counterarguments to wellness commodification", query="...")

for query in result.queries:
    content = exa_search(query)
    cleaned = content_cleaner(original_query=query, content=content)
```


This declarative style lets us focus on describing what we want the model to do. When processing a request about wellness commodification, the model might think:
1. Prioritize academic critiques over supporting literature
2. Look for papers from different theoretical frameworks
3. Focus on methodological challenges to similar analyses

Not included in the code, but worth noting: for the Exa query generation, I also provided examples and a prompting guide manually. While DSPy will infer this with a well defined metric and 10-100 examples, I had impactful examples from doing my task by hand already, so I passed them in.

The improvement pattern is straightforward. Every tool call should be paired with feedback:

```python
# Tool usage
result = search(purpose="Finding counterarguments to wellness commodification", query="...")
# note: tool desc will ask claude to provide feedback on the result

# result goes into conversation,
# claude provides feedback

feedback(
    tool_name="search",
    useful_output: bool,
    readability: Literal["perfect", "okay", "poor"],
    thoughts: str
)
```

This immediate feedback creates binary signal for improvement - was the output useful for the task or not? While getting feedback right after tool use means we might miss longer-term utility, this tradeoff gives us consistent, usable data from natural tool usage.

Braintrust makes this pattern practical by solving the logging challenge elegantly. Every tool call, every piece of feedback, every interaction gets automatically traced and stored in a format that's ready for optimization. Their key insight? Log straight to evals - making it trivial to turn your tool's actual usage into training data for DSPy.

The result is a natural improvement loop: your tool gets used, models provide feedback, Braintrust captures the interactions, and DSPy uses this data to make your tool better at its specific tasks. You can set it and forget it—check in on your tools after either you or your agents use them, evaluate, compile, optimize for quality, then do it again!

Pro tip: Start simple. Before diving into DSPy optimization, I'd recommend:
1. Start with basic prompt engineering in your chat client
2. Manually iterate until you get exactly the outputs you want
3. Use these successful examples in your DSPy modules

Auto-optimization needs a clear signal to work towards. If you can define metrics for your task, great! But many tasks are hard to quantify. While you can use an LLM as a judge, this requires careful thought about evaluation criteria and rubrics.

DSPy's simple signatures often work beautifully out of the box - something like "query, purpose -> list_of_exa_queries" might be all you need. But for more complex tasks, here's a practical shortcut: Use Anthropic's console to bootstrap your prompt from scratch, then use their 0-shot prompt optimizer. This often gets you 80% of the way there, especially for straightforward tasks. Even better, their prompt format matches DSPy's input-output structure, making it easy to transfer your work.

If DSPy still isn't giving you the results you want, understand what's actually going wrong: Does the model just not know about your task? Are you underspecifying what you actually want?

focus on the most critical stage in your pipeline. Remember: your modules are ultimately transformed into string prompts under the hood. The docstrings and descriptions aren't just documentation—they're telling DSPy exactly what you're trying to achieve at each step.


## What's Next?


MCP gives us standard building blocks for AI tools. I used it to cut my paper-writing time from 6 hours to 1 hour, but that's just scratching the surface. The real power is in how it lets us combine tools and improve them through use.

The patterns are simple:
- Design tools that match how LLMs think
- Let models provide feedback
- Use that feedback to make tools better

Between MCP's standardization and composability, DSPy's optimization capabilities, and tools like Braintrust making improvement dead simple, building powerful AI tools is now accessible to anyone willing to learn.

I'm [@dronathon](https://x.com/dronathon) on X - if you build something cool with these patterns, let me know! I'd love to see what you create.
