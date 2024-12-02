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

It's been an amazing respite because it let me do what I love without failing out of my classes.

A week ago, Anthropic dropped the Model Context Protocol (MCP), and now there's a standard way to build and connect all these tools. After two years of building AI tools, I've learned that context management and tool design matter more than anything else—and MCP finally gives us a standardized way to get both right.


## MCP: A New Standard for AI Integration

The Model Context Protocol (MCP) fundamentally changes how we can build AI tools. At its core, it provides:

1. A standardized way to expose **resources** - whether that's files, database records, or live system data. Your LLM can maintain awareness of available information without being overwhelmed by it. Usually larger documents.

2. **Tools** that let LLMs take actions through well-defined interfaces. Each tool has clear inputs, outputs, and error handling, making it easier to build reliable AI interactions.

What makes this exciting isn't just the individual capabilities - it's that it's an open, standard protocol. This means:
- Tools built by different developers can work together seamlessly
- Best practices can emerge and be shared across the community
- Integration patterns become consistent and reliable
- Many LLM applications will be able to connect to any MCP server to provide arbitrary functionality.

Instead of everyone building their own custom integration patterns, we get a common language for AI interactions. **MCP greatly reduces wasted effort.** My scattered collection of tools can now work together in a coherent system, and more importantly, they could work with tools built by others.


## Building tools for LLMs

Now that we have a standard way to connect tools, we can focus on making them actually good. Here's what I've learned about designing tools that LLMs love to use.

### Design
Over two years of building LLM tools, I've found one principle holds consistently: quality is directly proportional to how small the scope and how easily-parsed the inputs are. The tool needs to match how LLMs actually process information and tasks.

Think about what an LLM needs to effectively help with research. If you just ask it to "find papers about wellness," you're missing crucial context. What's the purpose of the research? What kind of analysis are you doing? What theoretical framework are you working within? Without this context, you'll get generic results that might be tangentially relevant but don't actually advance your work.

This led me to design around a simple pattern: **pair every query with its purpose.** Not just what you're looking for, but why you're looking for it. Here's what this looks like in practice:

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
        ),
        ExaQuery(
            text='Here is research on class dynamics in contemporary wellness culture:',
            category='research paper',
        ),
        ExaQuery(
            text="Here is a scholarly analysis of the wellness industry's impact on mental health:",
            category='research paper',
        ),
    ],
),
# category can be  'company', 'research paper', 'news', 'linkedin profile', 'github', 'tweet', 'movie', 'song', 'personal site', 'pdf'

# also can get live/recent results as well for time-sensitive queries
```

---


When you're brainstorming with Claude about a paper outline, you've come to a shared consensus about what you want to say, but you have to go through and do the difficult, time-consuming work of finding, analyzing, and understanding the data and literature to understand how your ideas interface with them. This conversation, where you go back the forth with the model as you flesh out your ideas, becomes valuable context for finding **exactly** the type of information you need.

The `(purpose, query)` pattern lets you capture this context precisely. Maybe you're looking for sources to:
- Back up your argument about wellness commodification
- Find counterarguments you haven't considered
- Explore how other scholars have approached similar analyses

Every time you search for something, you're not just telling the system what to look for—you're telling it why you need it in the context of your bigger project. This transforms every interaction from an isolated query into part of an ongoing conversation about your goals and needs.

When combined with Chain of Thought prompting, this understanding becomes even more powerful. Instead of just following instructions, the model actively thinks about the best way to help you—choosing better search strategies and finding more relevant sources. Because it knows both what you need and why you need it, it can also evaluate the results more intelligently, making sure you get exactly what you're looking for.

---

When building AI tools, it's easy to forget that your end user is actually the model itself. This means small design choices should focus on what makes sense to LLMs, not humans:
- XML-style outputs instead of human-readable formats
- Short, memorable IDs like "red-fish" instead of proper UUIDs
- Consistent patterns that match how models build internal representations

This becomes crucial when handling complex operations like codebase searches or package installations. Instead of dumping raw output, take time to structure what the model actually needs to see - relevant results, meaningful error messages, and clear success indicators. Building in this kind of thoughtful structure makes your tools more reliable and easier for models to use effectively.

But good design is just the start. The real magic happens when you let your tools improve through use.

### Building Tools That Learn


Here's something cool: you can build AI tools that get better through use. Not through complex retraining pipelines or massive datasets, but through simple, structured feedback from the models using them.

This is where DSPy comes in. DSPy provides a way to build modular AI systems that can optimize themselves. Instead of tinkering with prompts or managing complex training pipelines, DSPy lets you write clear Python code that describes what you want your LLM to do, then helps you make it better. By turning your interactions with the tool into training data, you make continuous improvement practically automatic.


For instance, my research assistant tool is built from these DSPy building blocks:

```python
class PurposeDrivenQuery(dspy.Signature):
    """Generate optimized search queries based on purpose and question."""

    purpose: str = dspy.InputField(
        desc='why do you want to know this thing? (ie: relevant context from your task)'
    )
    question: str = dspy.InputField(
        desc='what do you want to know, more specifically?'
    )
    queries: list[ExaQuery] = dspy.OutputField(
        desc='optimized queries following best practices (omitted)'
    )

class ExtractContent(dspy.Signature):
    """Extract and summarize relevant information from search results."""

    original_query: QueryRequest = dspy.InputField(
        desc='The query context for determining relevance'
    )
    content: SearchResultItem = dspy.InputField(
        desc='the raw search result'
    )
    cleaned_response: Union[SummarizedContent, None] = dspy.OutputField(
        desc='the cleaned and summarized result, `None` if no relevant content'
    )

```

These components become powerful when combined with DSPy's Chain of Thought capabilities. Instead of writing complex prompts like:

> "Think step by step about the search query. Show your reasoning in <thinking> tags. Consider the purpose and broader context first. Then generate 3-5 search queries formatted as JSON objects like `{"text": "...", "category": "..."}`. Make sure to include both general and specific queries. Format your final response as a list of queries, each on a new line starting with `-`. Remember to end each query with a colon and use natural language. If using categories, always include a non-category version too..."

We get much more than just structured reasoning. DSPy lets us specify our exact requirements in clean, maintainable Python code - no more massive `prompts.py` files or hunting through strings to update our pipelines. Just simple classes that clearly define what we want:

```python
query_generator = dspy.ChainOfThought(PurposeDrivenQuery)
content_cleaner = dspy.ChainOfThought(ExtractContent)

result = query_generator(purpose="Finding counterarguments to wellness commodification", query="...")

for query in result.queries:
    content = exa_search(query)
    cleaned = content_cleaner(original_query=query, content=content)
```


Because these components understand their purpose through Chain of Thought, they make intelligent decisions about how to handle each request. When working with our wellness commodification example, the query generator naturally:

- Prioritizes academic critiques over supporting literature
- Looks for papers from different theoretical frameworks
- Focuses on methodological challenges to similar analyses

While DSPy can learn these patterns from scratch with enough examples (usually 10-100), I found that feeding in high-quality examples from my manual research process made the tools immediately more effective.

But the real magic happens when these tools learn from every scenario they're used in. Whether you're searching documentation, installing packages, or finding research papers, each interaction is an opportunity for improvement. And here's the key: Claude itself provides the feedback, understanding the context of why the tool was called and whether its output actually helped:

```python
# Tool usage
result = search(purpose="Finding counterarguments to wellness commodification", query="...")
# note: tool desc asks claude to provide feedback on the result

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

Braintrust makes this pattern practical by solving the logging challenge elegantly. Every interaction gets automatically traced and stored, ready for optimization. The result is a natural improvement loop: your tool gets used, Claude provides feedback, Braintrust captures the data, and DSPy optimizes your tool's performance. You can set it and forget it—check in on your tools after either you or your agents use them, evaluate, compile, optimize for quality, then do it again!



## Getting Started


Understanding how DSPy works is crucial: your modules are ultimately transformed into prompts under the hood. This means:

- Docstrings and descriptions aren't just documentation—they're telling DSPy exactly what you want
- Clear input/output specifications often work beautifully. You can even express them inline with `dspy.ChainOfThought("query, purpose -> list_of_exa_queries")`
- These clean interfaces are what make optimization possible later - DSPy knows exactly what inputs and outputs to work with. Even better, you typically only need examples for your complete pipeline, not each individual component. DSPy handles the rest.

Here's a practical approach to building your first tools:

1. **Start Simple**
   - Use inline DSPy signatures if your task is simple enough.
   - If not, begin with basic prompt engineering in your chat client. Use Anthropics, it makes iterating and getting to okay quality much faster than OpenAI's playground. This guides you towards a clear, structured way of handling prompts that matches DSPy's abstractions surprisingly well.
   - This prompt engineering stage also helps you understand if an LLM can do the thing you want it to in just the one call. Is your scope too big? Will you need multiple calls?
   - Manually iterate until you get exactly the outputs you want
   - Use these successful examples in your DSPy modules (The library still needs a cleaner way of just putting examples into prompts, which I think might be possible to handle with a custom adapter)

2. **Build Incrementally**
   - Start with clear input/output specifications
   - Let DSPy's structure guide your tool design
   - Focus on getting the interfaces right - that's what enables optimization later
   - Keep your scope small and composable - chain simple tools rather than building complex ones

3. **Optimize Thoughtfully**
   - Focus on your critical pipeline stages
   - Keep your docstrings and descriptions precise—they're core to DSPy's understanding
   - When troubleshooting:
      - **READ YOUR LOGS.** Seriously.
      - **VISUALIZE YOUR PIPELINE.** Use tools like Arize Phoenix (an open-source OTEL provider) to see what's happening at each stage
      - Identify exactly where things break down. Focus on clarity: Does the model understand your task? Are your specifications clear?

The beauty of this approach? You can set it and forget it. Whether you or your agents are using the tools, they'll keep getting better through natural use.



## What's Next?


MCP gives us standard building blocks for AI tools. I cut my paper-writing time from 6 hours to 1 hour this semester, but that improvement took lots of manual iteration at first - copying queries, running searches, pasting results back to Claude. Now with MCP? That entire workflow is just one seamless step. And this example is just scratching the surface - the real power is in how MCP lets different tools work together. The power of each individual tool multiplies when they can improve themselves through use, automatically updating based on feedback from actual usage.


The patterns are simple:
- Design tools that match how LLMs think
- Let models provide feedback
- Use that feedback to make tools better

Between MCP's standardization and composability, DSPy's optimization capabilities, and tools like Braintrust making improvement dead simple, building powerful AI tools is now accessible to anyone willing to learn.

I'm [@dronathon](https://x.com/dronathon) on X - if you build something cool with these patterns, let me know! I'd love to see what you create.

Code coming soon! Still WIP because I'm not happy about latency, have to run git-filter-branch, and it's a big ol mess.
