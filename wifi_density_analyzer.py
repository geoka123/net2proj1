import pyshark
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# PHY type lookup table
phy_map = {
    '1': '802.11a',
    '2': '802.11b',
    '3': '802.11g',
    '4': '802.11n',
    '5': '802.11ac',
    '6': '802.11ax'
}

def export_wifi_data_to_csv(cap, filename, source_name):
    seen_bssids = set()
    records = []

    for packet in cap:
        try:
            ssid = '<hidden>'
            tag_info = packet[3].get('wlan.tag')
            if tag_info and 'SSID' in tag_info:
                ssid = tag_info.split(":")[2].strip().strip('"')

            transmitter = getattr(packet.wlan, 'ta', 'Unknown')
            bssid = getattr(packet.wlan, 'bssid', transmitter)

            if bssid in seen_bssids:
                continue
            seen_bssids.add(bssid)

            channel = getattr(packet.wlan_radio, 'channel', 'Unknown')
            freq = getattr(packet.wlan_radio, 'frequency', 'Unknown')
            signal = getattr(packet.radiotap, 'dbm_antsignal', 'N/A')
            noise = getattr(packet.radiotap, 'dbm_antnoise', None)
            phy_code = str(getattr(packet.wlan_radio, 'phy', ''))
            phy = phy_map.get(phy_code, f"Unknown ({phy_code})")

            signal_val = int(signal) if signal != 'N/A' else None
            noise_val = int(noise) if noise is not None else None
            snr = signal_val - noise_val if signal_val is not None and noise_val is not None else None

            records.append({
                'SSID': ssid,
                'BSSID': bssid,
                'Transmitter MAC': transmitter,
                'PHY Type': phy,
                'Channel': channel,
                'Frequency': f"{freq} MHz",
                'Signal Strength (dBm)': signal_val,
                'Noise (dBm)': noise_val,
                'SNR': snr,
                'Source': source_name
            })
        except Exception as e:
            print(f"Skipping packet due to error: {e}")
            continue

    df = pd.DataFrame(records)
    df.to_csv(f"{filename}.csv", index=False)
    print(f"✅ Exported {len(df)} unique beacons to {filename}.csv")
    return df

def plot_comparisons(df):
    # ---------- 1. SSID COUNT PER NETWORK ----------
    ssid_counts = df.groupby('Source')['SSID'].nunique()
    plt.figure(figsize=(6, 5))
    ssid_counts.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
    plt.title("Number of Unique SSIDs per Network")
    plt.ylabel("SSID Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plot_ssid_count_per_network.png", dpi=300)
    print("📊 Saved: plot_ssid_count_per_network.png")

    # ---------- 2. UNIQUE PHY TYPES PER NETWORK ----------
    unique_phy_counts = df.groupby('Source')['PHY Type'].nunique()
    plt.figure(figsize=(6, 5))
    unique_phy_counts.plot(kind='bar', color=['mediumseagreen', 'darkorange', 'cornflowerblue'])
    plt.title("Number of Unique PHY Types per Network")
    plt.ylabel("Unique PHY Types")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plot_unique_phy_types_per_network.png", dpi=300)
    print("📊 Saved: plot_unique_phy_types_per_network.png")

    # ---------- 3. SIGNAL AND NOISE BOXPLOTS ----------
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    boxplot_style = {
        'boxprops': dict(color='black', linewidth=2),
        'whiskerprops': dict(color='black', linewidth=2),
        'capprops': dict(color='black', linewidth=2),
        'medianprops': dict(color='red', linewidth=2),
        'flierprops': dict(marker='o', markerfacecolor='gray', markersize=5, linestyle='none')
    }

    # Signal Strength boxplot
    df_clean_signal = df[df['Signal Strength (dBm)'].notnull()]
    if not df_clean_signal.empty:
        df_clean_signal.boxplot(column='Signal Strength (dBm)', by='Source', ax=axs[0], **boxplot_style)
        axs[0].set_title('Signal Strength per Network')
        axs[0].set_ylabel('dBm')
        all_vals = df_clean_signal['Signal Strength (dBm)']
        ymin = int(np.floor(all_vals.min())) - 1
        ymax = int(np.ceil(all_vals.max())) + 1
        axs[0].set_yticks(np.arange(ymin, ymax + 1, 1))
        axs[0].grid(True, linestyle='--', alpha=0.6)
        medians = df_clean_signal.groupby('Source')['Signal Strength (dBm)'].median()
        for i, (source, median_val) in enumerate(medians.items()):
            axs[0].annotate(f"{median_val:.1f} dBm", xy=(i + 1, median_val), xytext=(0, -12),
                            textcoords='offset points', ha='center', fontsize=9, color='red')
    else:
        axs[0].text(0.5, 0.5, 'No signal data', ha='center', va='center')
        axs[0].set_title('Signal Strength per Network')
        axs[0].axis('off')

    # Noise boxplot
    df_clean_noise = df[df['Noise (dBm)'].notnull()]
    if not df_clean_noise.empty:
        df_clean_noise.boxplot(column='Noise (dBm)', by='Source', ax=axs[1], **boxplot_style)
        axs[1].set_title('Noise Level per Network')
        axs[1].set_ylabel('dBm')
        axs[1].grid(True, linestyle='--', alpha=0.6)
    else:
        axs[1].text(0.5, 0.5, 'No noise data', ha='center', va='center')
        axs[1].set_title('Noise Level per Network')
        axs[1].axis('off')

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("plot_signal_noise_comparison.png", dpi=300)
    print("📊 Saved: plot_signal_noise_comparison.png")

    # ---------- 4. SIGNAL STRENGTH PER SSID ----------
    plt.figure(figsize=(10, 6))
    df_ssid_signal = df[df['Signal Strength (dBm)'].notnull()]
    if not df_ssid_signal.empty:
        for source in df_ssid_signal['Source'].unique():
            subset = df_ssid_signal[df_ssid_signal['Source'] == source]
            plt.scatter(subset['SSID'], subset['Signal Strength (dBm)'], label=source, alpha=0.7)
        plt.title("Signal Strength of Each SSID per Network")
        plt.ylabel("Signal Strength (dBm)")
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Set y-axis ticks to finer granularity (e.g., 1 dBm steps)
        ymin = int(df_ssid_signal['Signal Strength (dBm)'].min()) - 1
        ymax = int(df_ssid_signal['Signal Strength (dBm)'].max()) + 1
        plt.yticks(np.arange(ymin, ymax + 1, 1))
        plt.savefig("plot_signal_strength_per_ssid.png", dpi=300)
        print("📊 Saved: plot_signal_strength_per_ssid.png")

# ------------------- MAIN -------------------
cap_home = pyshark.FileCapture('/home/geoka/tuc/net2/home_5g.pcapng', display_filter="wlan.fc.type_subtype == 8")
cap_home.close()
df_home = export_wifi_data_to_csv(cap_home, 'home_5g_data', 'home_5g')

cap_tuc = pyshark.FileCapture('/home/geoka/tuc/net2/tuc_5g.pcapng', display_filter="wlan.fc.type_subtype == 8")
cap_tuc.close()
df_tuc = export_wifi_data_to_csv(cap_tuc, 'tuc_5g_data', 'tuc_5g')

cap_home_2 = pyshark.FileCapture('/home/geoka/tuc/net2/home_2g.pcapng', display_filter="wlan.fc.type_subtype == 8")
cap_home_2.close()
df_home_2 = export_wifi_data_to_csv(cap_home_2, 'home_2g_data', 'home_2g')

combined_df = pd.concat([df_home, df_tuc, df_home_2], ignore_index=True)
plot_comparisons(combined_df)