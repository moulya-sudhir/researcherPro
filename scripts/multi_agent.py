from .searcher import get_top_n_documents, get_relevant_article_information
import autogen
import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()

class MultiLLMSystem():
    def __init__(self, data_path, vectorizer_path, tfidf_path, json_path, config_list_path, llm_config_path):
        self.data_path = data_path
        self.vectorizer_path = vectorizer_path
        self.tfidf_path = tfidf_path
        self.json_path = json_path
        config_list = autogen.config_list_from_json(
            env_or_file="configurations.json",
            file_location=config_list_path,
            filter_dict={
                "model": ["gpt-3.5-turbo-16k"],
            },
        )
        api_key = config_list[0]['api_key']
        openai.api_key = api_key
        self.llm_config = json.load(open(llm_config_path,'r'))
        self.llm_config['config_list'] = config_list
        self.llm_config_no_tools = llm_config_no_tools = {k: v for k, v in self.llm_config.items() if k != 'functions'}
    
    def get_documents(self, question, top_n = 3):
        # Get top n documents
        top_n_documents = get_top_n_documents(n=top_n, question=question, data_path=self.data_path, vectorizer_path=self.vectorizer_path, tfidf_path=self.tfidf_path)
        # Extract relevant information from top n documents
        relevant_articles_info = get_relevant_article_information(top_n_articles=top_n_documents, json_path=self.json_path)
        return relevant_articles_info
    
    def initialize_agents(self):
        self.user_proxy = autogen.UserProxyAgent(
            name='user_proxy',
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            system_message='''You should start the workflow by consulting the Gatherer, then the Reporter and finally the Moderator. 
            If the Gatherer does not use the `get_documents` function, you must request that it does.'''
        )

        self.gatherer = autogen.AssistantAgent(
            name="gatherer",
            system_message='''
            As the Information Gatherer, you must start by using the `get_documents` function to gather relevant data about the user's query. Follow these steps:

            1. Upon receiving a query, immediately invoke the `get_documents` function to collect all relevant article information related to the query. Do not proceed without completing this step.
            2. The `get_documents` function returns a Pandas DataFrame with each article containing information, specifically: title, abstract, limitation, result, and conclusion. Some columns may be missing for some articles and that is perfectly alright. 
            3. Present all of the article sections for each article in a structured format to the Reporter labelling each article and section, ensuring they have a comprehensive dataset to draft a response.
            4. Conclude your part with "INFORMATION GATHERING COMPLETE" to signal that you have finished collecting data and it is now ready for the Reporter to use in formulating the answer.

            Remember, you are responsible for information collection and retrieval only. The Reporter will rely on the accuracy and completeness of your findings to generate the final answer.

            ''',
            llm_config=self.llm_config,
        )

        self.reporter = autogen.AssistantAgent(
            name="reporter",
            system_message='''
            As the Reporter, you are responsible for formulating a detailed report of all the research gaps and future work that can be applied using the information provided by the Information Gatherer.

            1. Wait for the Information Gatherer to complete their task and present you with the relevant article information, which is a pandas DataFrame containing all article information like title, abstract, conclusion etc for all relevant articles.
            2. Using the gathered data, create a comprehensive and precise report of all the research gaps and future work that can be applied that adheres to the criteria of precision, depth, and clarity.
            3. Present your draft answer followed by "PLEASE REVIEW" for the Moderator to assess.

            If the Moderator approves your answer, respond with "TERMINATE" to signal the end of the interaction.

            If the Moderator rejects your answer:
            - Review their feedback.
            - Make necessary amendments.
            - Resubmit the revised answer with "PLEASE REVIEW."

            Ensure that your response is fully informed by the data provided and meets the established criteria.

            criteria are as follows:
            A. Precision: Properly identifies the research gaps and future work from the relevant article information.
            B. Depth: Provide comprehensive information using the data content.
            C. Clarity: Present information logically and coherently.
            ''',
            llm_config=self.llm_config_no_tools, 
        )

        self.moderator = autogen.AssistantAgent(
            name="moderator",
            system_message='''

            As the Moderator, your task is to review the Reporter's answers to ensure they meet the required criteria:

            - Assess the Reporter's answers after the "PLEASE REVIEW" prompt for alignment with the following criteria:
            A. Precision: Properly identifies the research gaps and future work from the relevant article information.
            B. Depth: Provide comprehensive information using the data content.
            C. Clarity: Present information logically and coherently.
            - Approve the answer by stating "The answer is approved" if it meets the criteria.
            - If the answer falls short, specify which criteria were not met and instruct the Reporter to revise the answer accordingly. Do not generate new content or answers yourself.

            Your role is crucial in ensuring that the final answer provided to the user is factually correct and meets all specified quality standards.

            ''',
            llm_config=self.llm_config_no_tools,
        )

        self.user_proxy.register_function(
            function_map={
                "get_documents": self.get_documents,
            }
        )

        self.groupchat = autogen.GroupChat(
            agents=[self.user_proxy, self.gatherer, self.reporter, self.moderator], 
            messages=[], 
            max_round=20
        )
        
        self.manager = autogen.GroupChatManager(
            groupchat=self.groupchat, 
            llm_config=self.llm_config, 
            system_message='''You should start the workflow by consulting the gatherer, 
            then the reporter and finally the moderator. 
            If the gatherer does not use the `get_documents` function, you must request that it does.'''
        )

    def start_conversation(self, question):
        self.manager.initiate_chat(
            self.manager, 
            message=question
        )