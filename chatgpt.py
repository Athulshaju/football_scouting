llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo") 
                chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
                result = llm(
                    chat_prompt.format_prompt(
                        player=choice, 
                        content=similar_players_df, 
                        style_t=game_styles[selected_game_style], 
                        type_player=player_experience[selected_player_experience], 
                        league=leagues[selected_league], 
                        ability=selected_player_skills, 
                        formation=selected_formation                    
                    ).to_messages()
                )
                
 
 
with col3:
                        st.caption("⚽ Information target of Player")
                        # GK
                        if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "GK":
                                col_1, col_2 = st.columns(2)

                                with col_1:
                                    st.metric("Saves", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Saves'].iloc[0], None, help="Total number of saves made by the goalkeeper.")
                                    st.metric("Clean Sheet", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance CS'].iloc[0], None, help="Total number of clean sheets (matches without conceding goals) by the goalkeeper.")

                                with col_2:
                                    st.metric("Goals Against", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance GA'].iloc[0], None, help="Total number of goals conceded by the goalkeeper.")
                                    st.metric("ShoTA", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance SoTA'].iloc[0], None, help="Total number of shots on target faced by the goalkeeper.")

                        # DF
                        if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "DF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "DF,MF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "DF,FW":
                            col_1, col_2, col_3 = st.columns(3)

                            with col_1:
                                st.metric("Assist", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the defender.")
                                st.metric("Goals", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the defender.")

                            with col_2:
                                st.metric("Aerial Duel", df_player.loc[df_player['Player'] == ultimo_nome, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the defender.")
                                st.metric("Tackle", df_player.loc[df_player['Player'] == ultimo_nome, 'Tackles TklW'].iloc[0], None, help="Total number of successful tackles made by the defender in 2022/2023.")

                            with col_3:
                                st.metric("Interception", df_player.loc[df_player['Player'] == ultimo_nome, 'Int'].iloc[0], None, help="Total number of interceptions made by the defender.")
                                st.metric("Key Passage", df_player.loc[df_player['Player'] == ultimo_nome, 'KP'].iloc[0], None, help="Total number of key passes made by the defender.")

                        # MF
                        if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "MF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "MF,DF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "MF,FW":
                            col_1, col_2, col_3 = st.columns(3)

                            with col_1:
                                st.metric("Assist", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                                st.metric("Goals", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                                st.metric("Aerial Duel", df_player.loc[df_player['Player'] == ultimo_nome, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                            with col_2:
                                st.metric("GCA", df_player.loc[df_player['Player'] == ultimo_nome, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                                st.metric("Progressive PrgP", df_player.loc[df_player['Player'] == ultimo_nome, 'Progression PrgP'].iloc[0], None, help="Total number of progressive passes by the player.")

                            with col_3:
                                st.metric("SCA", df_player.loc[df_player['Player'] == ultimo_nome, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                                st.metric("Key Passage", df_player.loc[df_player['Player'] == ultimo_nome, 'KP'].iloc[0], None, help="Total number of key passes by the player.")

                        # FW
                        if df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "FW" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "FW,MF" or df_player.loc[df_player['Player'] == ultimo_nome, 'Pos'].iloc[0] == "FW,DF":
                            col_1, col_2, col_3 = st.columns(3) 

                            with col_1:
                                st.metric("Assist", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Ast'].iloc[0], None, help="Total number of assists provided by the player.")
                                st.metric("Goals", df_player.loc[df_player['Player'] == ultimo_nome, 'Performance Gls'].iloc[0], None, help="Total number of goals scored by the player.")
                                st.metric("Aerial Duel", df_player.loc[df_player['Player'] == ultimo_nome, 'Aerial Duels Won'].iloc[0], None, help="Percentage of aerial duels won by the player.")

                            with col_2:
                                st.metric("SCA", df_player.loc[df_player['Player'] == ultimo_nome, 'SCA SCA'].iloc[0], None, help="Total number of shot-creating actions by the player.")
                                st.metric("xG", df_player.loc[df_player['Player'] == ultimo_nome, 'Expected xG'].iloc[0], None, help="Expected goals (xG) by the player.")
                                st.metric("xAG", df_player.loc[df_player['Player'] == ultimo_nome, 'Expected xAG'].iloc[0], None, help="Expected assists (xAG) by the player.")

                            with col_3:
                                st.metric("GCA", df_player.loc[df_player['Player'] == ultimo_nome, 'GCA GCA'].iloc[0], None, help="Total number of goal-creating actions by the player.")
                                st.metric("Key Passage", df_player.loc[df_player['Player'] == ultimo_nome, 'KP'].iloc[0], None, help="Total number of key passes by the player.")                