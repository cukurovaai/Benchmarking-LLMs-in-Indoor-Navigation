import os
import openai
import re

def read_system_prompt(goal):
    system_prompt = None

    if goal == "object":
        with open("/Bench_LLM_Nav/prompts/object_system_prompt.txt", 'r') as f:
            system_prompt = f.read()
    elif goal == "spatial":
        with open("/Bench_LLM_Nav/prompts/spatial_system_prompt.txt", 'r') as f:
            system_prompt = f.read()
    elif goal == "cot":
        with open("/Bench_LLM_Nav/prompts/cot_spatial_system_prompt.txt", 'r') as f:
            system_prompt = f.read()
    elif goal == "reasoning":
        with open("/Bench_LLM_Nav/prompts/reasoning_prompt.txt", 'r') as f:
            system_prompt = f.read()
    return system_prompt

def extract_python_code(content):
    code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        #print(full_code)
        return full_code
    else:
        return None

def parse_object_goal_instruction(language_instr, technique):
    """
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
    import openai

    system_prompt = read_system_prompt("object")
    openai_key = os.environ["LLM_KEY"]
    openai.api_key = openai_key
    client = openai.OpenAI(api_key=openai_key)

    if technique == 'few-shot' or technique == 'reasoning':
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "user", "content": system_prompt},
                    {"role": "user", "content": "go to the kitchen and then go to the toilet"},
                    {"role": "assistant", "content": "kitchen, toilet"},
                    {"role": "user", "content": "go to the chair and then go to another chair"},
                    {"role": "assistant", "content": "chair, chair"},
                    {"role": "user", "content": "navigate to the green sofa and turn right and find several chairs, finally go to the painting"},
                    {"role": "assistant", "content": "green sofa, chairs, painting"},
                    {"role": "user", "content": "approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen"},
                    {"role": "assistant", "content": "window, television, oven, kitchen"},
                    {"role": "user", "content": "walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed"},
                    {"role": "assistant", "content": "plant, table, bedroom, bed"},
                    {"role": "user", "content": "go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room"},
                    {"role": "assistant", "content": "stairs, room, book shelf, table, next room"},
                    {"role": "user", "content": "Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances"},
                    {"role": "assistant", "content": "table, cushion, column, appliances"},
                    {"role": "user", "content": "Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table"},
                    {"role": "assistant", "content": "chair, table"},
                    {"role": "user", "content": language_instr}
                ],
                max_tokens=300,
            )
    elif technique == 'zero-shot':
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": language_instr}
                ],
                max_tokens=300,
            )
    else:
        print("Invalid technique")
        return None

    text = response.choices[0].message.content
    return [x.strip() for x in text.split(",")]

def parse_spatial_goal_instruction(language_instr, technique):
    import openai
    
    if technique.startswith("reasoning"):
        system_prompt = read_system_prompt("reasoning")
    else:
        system_prompt = read_system_prompt("spatial")
    
    openai_key = os.environ["LLM_KEY"]
    openai.api_key = openai_key
    # instructions_list = language_instr.split(",")
    instructions_list = [language_instr]
    
    model = "gpt-4o"
    results = ""
    for lang in instructions_list:
        client = openai.OpenAI(api_key=openai_key)
        # if technique == "few-shot" or technique == 'reasoning':
        response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "robot.turn_absolute(-90)\nrobot.move_forward(3)"},
            {"role": "user", "content": lang},
            ],
            max_tokens=300,
        )
        """
        elif technique == "zero-shot" or technique == "chain-of-thought":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": system_prompt},
                    {"role": "user", "content": lang},
                ],
                max_tokens=300,
            )
        else:
            print("Invalid technique")
            return None
        """

        print(f"Model: {model}")
        print(f"System prompt: {system_prompt}")
        text = response.choices[0].message.content
        text = extract_python_code(text)
    return text


if __name__ == '__main__':
    text = parse_spatial_instruction("go to the sofa, turn right and move in between the table and the chair, and then move back and forth to the keyboard and the screen twice", "zero-shot")
    print(text)
